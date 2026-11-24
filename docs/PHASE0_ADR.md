# Phase 0 — Environment validation & architecture decisions

Date: 2026-11-24 · Status: **Proposed (awaiting approval of blockers B1–B3)**

Everything in "Observed" was actually run on this machine. Everything in "Unverified" has not been run yet and must not be claimed as working.

---

## 0. Two-machine model (decided 2026-11-24)

| Machine | Role | Consequence |
|---|---|---|
| **Dev laptop** (this PC: i5-1035G1, Intel UHD, no NVIDIA) | Write code, run automated tests, commit | The Streamlit app is never run here for real work. Hardware detection results from this PC are irrelevant to the product |
| **Run laptop** (NVIDIA RTX 4050, 6 GB VRAM) | Fresh `git clone`, run `setup_windows.bat`, run the app | Primary runtime target. GPU paths can only be verified there |

Rules that follow from this:
- The repo must be fully portable: no absolute paths, no machine-specific config committed, `.env` / DB / media / venv git-ignored.
- `setup_windows.bat` and `scripts/verify_installation.py` on the run laptop are the **only** place GPU capability is decided. Nothing GPU-related is assumed at code-writing time.
- The CPU fallback is still required and remains the path exercised by automated tests on the dev laptop.
- Every GPU-specific behaviour is reported as "implemented, unverified" until you run `scripts/diagnostics.py` on the run laptop and share the report.
- Gemini keys, Ollama models and YouTube OAuth secrets are configured on the run laptop only.

## 1. Observed environment (dev laptop)

| Item | Result | Effect |
|---|---|---|
| OS | Windows 11 Pro 10.0.26200 | Matches target |
| CPU / RAM | Intel i5-1035G1 (4 cores / 8 threads), 19.8 GB | CPU-only path is the primary test target |
| GPU | Intel UHD Graphics only. No `nvidia-smi` | Hardware mode = `CPU`. **The NVIDIA path cannot be tested on this PC** |
| Python | Only **3.14.7** (`py -0p` lists no other version) | **Blocker B1**: the spec allows only 3.11/3.12 |
| FFmpeg / FFprobe | **Not found** on PATH or in WinGet package folders | **Blocker B2**: core requirement |
| winget | Available | Can install Python 3.12 and FFmpeg (needs your OK) |
| Git | 2.55.0 | OK. The working folder is not a git repo yet |
| Disk free | D: 907 GB, C: 67 GB | OK. Keep media and the venv on D: |
| Ollama | 0.40.16, running on `localhost:11434` | OK |
| Ollama models | `qwen2.5:3b` (Q4_K_M). Capabilities reported: `completion, tools`, **no `vision`** | Local visual AI **not available** until a vision model is installed (**B3**) |
| uv | Not installed | Not needed. Use stdlib `venv` + pip |

## 2. Dependency validation (cp312 / win_amd64 wheels, actually downloaded)

Command: `pip download --python-version 3.12 --platform win_amd64 --only-binary=:all: --no-deps <pkg>`

| Package | Wheel resolved | Decision |
|---|---|---|
| streamlit | 1.67.0 (Requires-Python ≥3.11) | **Core** |
| pydantic | 2.14.4 | **Core** |
| faster-whisper | 1.2.1. Requires `ctranslate2<5,>=4`, `onnxruntime<2,>=1.14`, `av>=11`, `tokenizers`, `huggingface-hub` | **Core** |
| ctranslate2 | 4.9.1 cp312 | Core (pulled in by faster-whisper) |
| av (PyAV) | 19.0.2 abi3 | Core (pulled in) |
| onnxruntime | 1.32.0 cp312 | **Core, the only ONNX runtime** (see ADR-4) |
| onnxruntime-gpu | 1.32.0 cp312 | **Not used** (see ADR-4) |
| numpy | 2.5.5 | Core |
| pillow | 13.0.0 | Core |
| opencv-python-headless | 5.1.0.95 abi3 | Core (frame difference, sharpness, contact sheets) |
| soundfile | 0.14.0 | Core (WAV I/O for audio features) |
| google-genai | 3.6.1 | Core, but only called when the user opts in |
| httpx | 0.28.1 | Core (Ollama HTTP client) |
| python-dotenv | 1.2.4 | Core |
| pytest | 9.2.0 | Dev |
| google-api-python-client / google-auth-oauthlib | 2.202.0 / 1.5.0 | Optional (YouTube private upload) |
| rembg | 2.0.87 (pulls scikit-image, scipy, pymatting, pooch) | Optional (`requirements-optional.txt`) |
| scenedetect | 0.7.3 | **Not adopted**: OpenCV frame-diff covers the need with one dependency fewer |
| auto-editor | 29.3.1, a `py3-none-any` wheel that declares **no** dependencies | Optional. CLI syntax unverified; gated behind a runtime version/CLI check |
| **pycaps** | **No package of that name on PyPI**. A web search also found nothing | **Not adopted.** The FFmpeg `.ass` path becomes the primary caption renderer. A detector for an installed `pycaps` is left out rather than shipped as an untested stub |

Lock strategy: after B1/B2 are resolved, install into a fresh 3.12 venv, run the import and smoke tests, then freeze **exact** pins into `requirements.txt` (tested set). `requirements-optional.txt` holds rembg, auto-editor, and the YouTube client. No `requirements-gpu.txt` is needed for ONNX (ADR-4). CUDA for Whisper comes from CTranslate2 plus NVIDIA's cuBLAS/cuDNN runtime. That GPU file will list only the documented NVIDIA runtime wheels, **marked untested** until it runs on the RTX 4050.

## 3. Gemini facts (from official docs, fetched 2026-11-24)

Docs now redirect to `aistudio.google.com/docs/...`.

- `gemini-3.8-flash` exists. Model page: "Latest update: September 2026". Inputs: text, image, video, audio, PDF. Output: text. 1,048,576 input tokens.
- The pricing page lists a **Free Tier, "Free of charge"** input/output for `gemini-3.8-flash`, `gemini-3.6-flash`, and `gemini-3.5-flash-lite`. `gemini-3.7-flash` is deprecated and auto-routes to 3.8.
- **Free-tier rate limits for these models are not published on the rate-limits page.** It says limits are "not guaranteed" and are applied per project. RPD resets at midnight Pacific. → The app must learn limits empirically: handle HTTP 429, back off, then pause the job or fall back to Ollama.
- **Privacy:** the free tier is marked "Used to improve our products: Yes". **This must appear in the cloud-upload consent dialog.**
- The model page does not mention the Files API. → Check upload support at runtime in the connection test. Fall back to sampled frames sent inline.
- Paid prices for 3.8 Flash are scheduled to rise on 2027-01-01. Irrelevant at ₹0, but a reminder that the app must never enable billing.
- Default model setting: `gemini-3.8-flash`, editable. The app validates it against `models.list` with the user's key before use. **I cannot run that check without a key**, and none is configured on this PC.

## 4. Architecture decisions

**ADR-1 Python 3.12.x only.** It is the newest allowed version, and every core wheel resolves for cp312. 3.11 will not be claimed as compatible because it won't be tested.

**ADR-2 Repository root = `D:\YTAnalyzer`** (the current folder), laid out as in spec §4. No extra `yt_creator_studio/` nesting unless you want it.

**ADR-3 FFmpeg is external**, installed via winget (`Gyan.FFmpeg`) or by the user. Required features are checked at runtime, and each missing one disables only the feature that needs it: `sidechaincompress`, `amix`, `loudnorm`, `silencedetect`, `ass`/libass, `boxblur`, `libx264`, `h264_nvenc` (plus a real encode test).

**ADR-4 Single ONNX runtime = CPU `onnxruntime`.** faster-whisper hard-requires it (Silero VAD). Whisper's CUDA path runs through CTranslate2, not ONNX. So the GPU environment needs no `onnxruntime-gpu`, and two runtimes never end up in one venv. rembg runs on CPU, which is acceptable for 5 frames.

**ADR-5 Hardware mode** is computed by real tests in this order: nvidia-smi → NVENC listed → NVENC 1-second encode → CTranslate2 CUDA device count → faster-whisper tiny CUDA transcription of a fixture. On this PC the expected result is `CPU`. GPU code paths will be implemented and unit-tested with mocks, and labeled **untested on hardware**.

**ADR-6 Whisper defaults:** CPU → `small` with `int8`, chosen for quality. `medium` and `large-v3-turbo` are selectable with a speed/RAM warning. CUDA → picked by available VRAM. On OOM or CUDA errors the app retries on CPU, and the job report shows the effective model.

**ADR-7 Local AI:** `qwen2.5:3b` is text-only, so it is used for transcript/event reasoning and metadata only. The UI states that local visual understanding is unavailable. Visual evidence then comes from computed signals (motion, scene change, sharpness, audio) plus user review. A local vision model is used only if you install one and the `/api/show` capabilities include `vision`.

**ADR-8 Captions:** I write ASS files directly (no extra dependency) and burn them in with FFmpeg `subtitles`/`ass`. Word-level karaoke styling comes from faster-whisper word timestamps.

**ADR-9 Silence/inactivity:** native implementation using `silencedetect`, RMS, and OpenCV frame difference. Strong events always override cuts. The no-cut mode is the default. auto-editor is optional and used only after its CLI passes a check.

**ADR-10 Timeline interchange:** export FCP7 XML (`xmeml`) and a CMX3600 EDL. **Resolve import cannot be verified here** (Resolve is not installed). Both formats ship labeled "unverified" until imported into a named Resolve version.

**ADR-11 YouTube:** official `google-api-python-client` `videos.insert` with resumable `MediaFileUpload`, scope `youtube.upload`, and `privacyStatus=private` hard default. The result is described as a **private video**, not a "draft". Upload quota cost is to be verified in Google's current docs.

**ADR-12 Persistence:** SQLite with a `schema_version` table and ordered migration functions. WAL mode. Parameterized SQL only. Stage fingerprint = SHA-256 of (input file checksums + relevant settings JSON + stage code version).

**ADR-13 Concurrency:** long jobs run in a background worker thread/process owned by `job_manager`, not inside Streamlit callbacks. Streamlit polls the DB for progress. Cancellation is a DB flag checked between chunks.

## 4b. Dev-laptop verification results (run 2026-11-24)

Installed for **testing only**: Python 3.12.10 (winget, user scope) and FFmpeg 9.1 full build (gyan.dev, via winget; the first attempt failed with a file-copy "Access is denied", and the retry succeeded). Test venv: `D:\YTAnalyzer\.venv` (git-ignored).

| Check | Result |
|---|---|
| FFmpeg filters: sidechaincompress, amix, loudnorm, silencedetect, ass, subtitles, boxblur, gblur, scale, crop, overlay, vstack, highpass, afftdn, alimiter, afade, ebur128, astats, select, tile, asplit, volume, adelay, aresample, pan | All present |
| Encoders: libx264, aac, pcm_s16le, libmp3lame, h264_nvenc, hevc_nvenc | All **listed** |
| libass | Enabled |
| Real NVENC encode | **Failed**: "Cannot load nvcuda.dll" (no NVIDIA GPU). Confirms that "listed" ≠ "usable" and that the real encode test is necessary |
| Generated 1080p clip with 3 audio tracks, path containing a space and `ü` | Created and probed correctly (stream indexes 1–3) |
| Ducking `sidechaincompress` (game a:1, voice a:2 as sidechain) | Game level −21.1 dB → **−27.4 dB while voice active** → back to −21.1 dB |
| 9:16 blurred-background layout + ASS burn-in (`\k` karaoke tags) | 1080×1920, 6.0 s, audio present; caption visually confirmed in an extracted frame |
| Python imports (3.12.10) | streamlit 1.67.0, pydantic 2.14.4, faster-whisper 1.2.1, ctranslate2 4.9.1, av 19.0.2, onnxruntime 1.32.0, numpy 2.5.5, pillow 13.0.0, opencv 5.1.0, soundfile 0.14.0, google-genai 3.6.1, httpx 0.28.1, pytest 9.2.0, google-api-python-client, google-auth-oauthlib: all OK |
| ONNX Runtime providers | Azure, CPU |
| CTranslate2 CUDA device count | 0 (expected on this laptop) |
| Streamlit APIs | `st.audio_input`, `st.fragment`, `st.dialog` exist (in-browser voiceover recording is possible) |
| google-genai | `genai.Client`, `types.GenerateContentConfig` exist |
| Ollama HTTP API | Reachable, version 0.40.16 |

Not yet run: a faster-whisper transcription (needs a model download from Hugging Face, ~75 MB for `tiny`), any Gemini call (no key, by design), a YouTube upload, and anything on the run laptop.

Run-laptop GPU note (**unverified**): CTranslate2 4.x CUDA builds need a matching CUDA 12 cuBLAS + cuDNN 9 runtime. The installer will try NVIDIA's pip runtime wheels and fall back to CPU if the faster-whisper CUDA smoke test fails. Exact versions get pinned only after a successful run there.

## 5. Open decisions

- **B1/B2: resolved.** Python 3.12.10 and FFmpeg 9.1 are installed on the dev laptop for testing only (see 4b).
- **B3 (run laptop):** a local vision model for Ollama is a run-laptop decision. With 6 GB VRAM a small vision model may fit alongside nothing else; the exact tag must be checked against the current Ollama library before recommending one. Never auto-downloaded.
- **Run-laptop facts that would help (optional):** output of `nvidia-smi` (driver + CUDA version) and the Python version installed there. The installer detects these itself, so this is not blocking.
- Whether you use DaVinci Resolve, and which version, for ADR-10.

## 6. Implementation plan (after approval)

Phases 1–6 as in the spec. Each ends with a list of files changed, real `pytest` output, and known limitations.

1. Foundation: config/paths, SQLite + migrations, logging, job state machine, installer, diagnostics, import + FFprobe, track mapping.
2. Media engine: extraction, 3-way mix + ducking, voiceover, transcription, silence/inactivity report, rough cut, EDL/XML.
3. Intelligence: windows, frame sampling, audio features, provider layer (Gemini mocked in tests), fusion, scoring, candidates, review UI.
4. Content: Shorts (3 layouts), ASS captions, thumbnails, creator pack.
5. Review/export: approval checklist, per-output re-render, private upload.
6. Hardening: synthetic fixture media made with FFmpeg (1/2/3/0 audio tracks, VFR, silence + quiet scene), resume tests, README.

Known now as **untestable on this PC**: NVENC, CUDA Whisper, Resolve import, live Gemini calls (no key), real YouTube upload (no OAuth client).
