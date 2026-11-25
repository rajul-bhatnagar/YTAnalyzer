# YT Creator Studio

Local-first Windows 11 app that turns PC gameplay recordings into a reviewed YouTube content package.

> **Status: all 6 phases implemented on the dev PC (CPU).** The full workflow works: import → audio → analysis →
> highlights → Shorts → thumbnails → creator package → final approval → optional private YouTube upload. What is
> **not yet verified** needs your hardware or accounts (RTX GPU, Gemini key, YouTube account, Resolve); see
> [Tested vs. untested](#tested-vs-untested-as-of-phase-4) and [docs/MIGRATION_RTX4050.md](docs/MIGRATION_RTX4050.md).
>
> Phase 6 added:
> - **Temporary-file clean-up** (Settings): previews, proxy chunks, Short work folders and orphaned partial files. It lists only regenerable files, asks for confirmation, refuses anything else, and doesn't run while a job does.
> - **Low-disk warnings**, with processing refused below 2 GB.
> - **Sidebar quick settings.**
> - The migration guide, a diagnostics report from the dev PC (`docs/diagnostics_dev_pc.json`), and a sample project manifest made from synthetic media (`docs/sample_manifest.json`, from `scripts/make_sample_project.py`).

## Upgrade: style profiles, long-recording gating, Short polish (2026-11-25)

- **Style profiles** (game-agnostic, by content style): each style sets default Shorts layout, dead-air handling (trim / speed up / none; none for Horror and Story) and HUD-text weight (0 for Horror and Story). Override any of them per project on the Shorts page; an empty override means "use the profile".
- **Analysis windows overlap by 30 s**, so a moment on a window boundary is seen whole; duplicates are merged.
- **AI request cap for long recordings:** windows are ranked by local measurements (sound jumps, cuts, motion, voice, HUD-text changes). Only the top **12** (adjustable), plus time ranges you pin, are sent to Gemini or Ollama. The rest are marked "skipped", not failed. Finished windows are always kept.
- **Per-Short fact-check:** the hook and title are checked against the evidence *inside the clip's time range*. Words or numbers from elsewhere in the recording, or numbers found nowhere, are flagged, with one-click in-clip suggestions. Flags must be acknowledged at final approval.
- **Dead air per Short:** static, quiet lead-in is trimmed or played at 2–2.5×, and a static tail is trimmed. At least 2 s are kept before the key moment and 1.5 s after the last activity, and a Short never drops below 15 s. Captions move with the edits. The long video is never changed.
- **Split layout without a facecam:** the bottom panel can be any shape (e.g. a thin full-width score strip, sized automatically, never stretched). Choosing "whole frame" there shows a warning.
- **Safe-zone overlay** in the framing preview: the Shorts player's usual UI areas (bottom 22%, right 12%; approximate).
- **Voiceover quick loop** on the Shorts and Metadata pages: upload or record, then "Apply". This re-mixes (ducked, loudness-normalised), re-transcribes, re-renders approved Shorts and updates the package, without repeating import, mapping, analysis or highlights. A Short can also have its own voiceover.
- **Optional HUD text reading (OCR):** if `rapidocr-onnxruntime` is installed, digit changes in top and bottom HUD bands become a bonus signal and ground-truth text for the AI and fact-checks. Clock-like bands are ignored, and empty results never lower a score. **Not installed by default and untested here:**
  ```powershell
  .\.venv\Scripts\python.exe -m pip install rapidocr-onnxruntime
  ```
  Re-run `scripts\verify_installation.py` afterwards, since pip may change the pinned `onnxruntime`.
- **Thumbnails:**
  - each highlight has an **outcome mood** (win_hype, clutch_tension, fail_shock, funny_chaos);
  - Gemini prompts are **text-free** and styled by mood;
  - a local **composer** adds 2–3 bold uppercase words (rejecting phrases that end in a preposition) and keeps the bottom-right clear;
  - the safe-area preview can show a mock video-length badge.
- **Hardening:**
  - **A/V sync fix:** when a recording's video starts after its audio (common on phones), every cut now accounts for the gap. It was measured at about 0.5 s of drift before the fix and under 0.05 s after.
  - Shorts always encode stereo AAC: FFmpeg 9.1 stalled on mono input at 192 kb/s.
  - FCP7 XML and EDL use `file://localhost/D:/…` URLs.
  - GPU memory is freed before a local vision model runs, and Ollama models are unloaded (`keep_alive: 0`) after analysis and before GPU transcription.
  - Final approval shows non-blocking YouTube quota and testing-mode sign-in notes.

## Phase 5 (final approval & upload)

- **Final approval & upload** page: a checklist that re-checks every approved file against its validation record.
  - It **blocks** when there's nothing to publish, an approved file is missing, changed or failed, or [placeholders] remain in the title or description.
  - Flagged numbers and uncertain lines need your explicit acknowledgement.
  - Approval writes `final_manifest.json` (paths, sizes, checksums). If an approved file's size changes later, it must be approved again.
- **Optional YouTube upload** uses the official API (`youtube.upload` permission only, resumable upload, retries on server errors), and the field names were checked against the API document bundled with the client. Safeguards:
  - **private by default**; public or unlisted needs an extra confirmation;
  - you must answer "made for kids" yourself;
  - the token is stored in the git-ignored `data/` folder;
  - quota or sign-in errors are reported without touching your approval.

  The result is a **private video** on your channel, not a YouTube Studio draft. The custom thumbnail is set only for the long video, and that may require a verified channel. **Not tested with a real account**: on the run PC, set `YOUTUBE_CLIENT_SECRETS_FILE` (an OAuth "Desktop app" client JSON kept outside the repo) and use **Connect YouTube account**. Category 20 ("Gaming") is the app's assumption; verify it.

## What works now: Phase 4 (Shorts, thumbnails, metadata)

- **Shorts** (Shorts page): each **approved** highlight becomes a 1080×1920 Short, cut from the original recording with your chosen audio mix. Three layouts, none of which stretch the picture:
  - full frame over a blurred background (the whole HUD stays visible);
  - an adjustable vertical crop;
  - a split stack: zoomed action on top, the whole frame or a chosen region (e.g. a scoreboard) below.

  Before rendering you can check the framing on a single frame, and adjust trim times and the hook.
- **Captions**: word-by-word ASS captions burned in with FFmpeg/libass. The spoken word is highlighted, and font size, colours, outline, optional box and position (with an adjustable bottom margin for the Shorts UI) are all configurable. A short hook caption can show when nobody speaks. pycaps is not used (no such package on PyPI).
- **Only what changed re-renders**: each Short is its own stage. Changing one Short's layout re-renders only that Short, and the audio mix is reused. Previous renders go to `_previous/`. Title or description edits never re-render video.
- **Validation of every Short**:
  - 1080×1920 size, duration, decodable start and end;
  - audio present and not silent;
  - caption timing inside the clip;
  - **black-frame detection** (output black while the source isn't = a filter error).

  You can approve a Short only after it passes.
- **Thumbnails** (Thumbnails page):
  - **5 candidate PNG frames** at full resolution from your approved highlights (key moment, just before and after, the clip middle, AI-cited frames; never loudness alone), ranked by sharpness, exposure and colour, with near-duplicates removed;
  - two ready-to-copy prompts for the free Gemini web app (edit-a-screenshot, and from-scratch 16:9), built only from what happens in your video, with ≤ 3 uppercase words and the bottom-right corner kept clear;
  - import of your finished thumbnail, with size/ratio checks and a timestamp-safe-area preview.

  No image is generated by the app and no paid image API is called.
- **Creator package** (Metadata & scripts page), saved as `youtube_pack.json` and `youtube_pack.txt`:
  - 5 titles, a ~20 s opening hook, and a timestamped voiceover script (recording time and rough-cut time);
  - a title and hook per Short, a description, 10 tags and a pinned comment;
  - both thumbnail prompts, a style rationale, and "check before publishing" notes.

  It's written by your AI provider from the evidence only (approved highlights, events, transcript). If there's no working AI, you get an English template with **[placeholders]**, never invented text.
- **Guards against invented facts**:
  - **any number not found in the evidence is flagged** (e.g. an invented "99-run chase");
  - script lines can't describe an event before it happens;
  - unknown event references are marked for checking;
  - your own highlight titles and hooks always win.
- **Edits are kept**: every field is editable and saved per field. "Generate / update" keeps your edits; discarding them needs an explicit tick.

## Phase 3 (analysis & highlights)

On the **Analysis & highlights** page:

- **Measured evidence (always, no AI needed):**
  - sudden game sounds and loudness jumps, measured against the local level;
  - emphatic voice, compared with the speaker's own level in other segments (needs ≥ 3 segments and ≥ 20 s of speech);
  - hard scene changes and sustained motion;
  - possible audio clipping;
  - exact transcript quotes.

  Signals are clustered into timestamped events. A moment supported by only one kind of signal is flagged, e.g. "Audio peak only: could be music, an effect or clipping".
- **Analysis windows** (default 3 min) with frames sampled from the **original** recording: a regular grid plus extra frames around each measured event, with sharpness, black-frame and near-duplicate checks.
- **AI providers** (one interface; you choose):
  - **Ollama** (local): uses models you installed yourself, and never downloads one. A model with `vision` in Ollama's capabilities gets labelled contact sheets; a text-only model gets the transcript and measurements and is clearly marked as unable to see.
  - **Gemini** (cloud, free tier): the official `google-genai` SDK, **opt-in per project** with a disclosure of exactly what is sent. Sends sampled frames by default; low-resolution video chunks of the most active windows need a separate approval, and uploads are deleted after use.
  - **None**: measured signals only.
- **Strict validation of every AI answer** (Pydantic + checks against the evidence):
  - frame times must match frames that were actually sent;
  - times are clamped to the window, and events outside it are dropped;
  - **a transcript quote must really be in the transcript, or it is removed**;
  - a text-only model's visual claims are removed.

  One repair request is made for malformed output; if that also fails, the window is marked for your review.
- **Quota handling**: on a Gemini 429 the app switches to local Ollama or pauses the job (your choice). Each window's result is cached, so **Resume continues where it stopped**. Requests are paced and capped per job.
- **Explainable score** (0–100, rounded to 5, with a band: strong / promising / possible / weak):
  - 8 AI-rated factors plus 4 measured ones, weighted by content style; the style changes emphasis, never the evidence standard;
  - factors the AI could not judge are **excluded, not counted as zero**;
  - moments with no AI description are **capped at 70**, because measurements show where something happens, not whether it is good.
- **Shorts candidates** (default 3, 20–50 s, configurable): context before the key moment and aftermath after it, cut points snapped to calm picture, **never cutting mid-sentence**, within the recording, with ±5 s trimming handles. Duplicates of the same moment are merged.
- **Review**: for every candidate you get:
  - a 480p preview cut from the original;
  - the rationale, the uncertainty, and evidence (frames with observations, audio notes, transcript quotes, AI provider/model/confidence);
  - the factor breakdown;
  - editable start/end, hook and title, with validation and warnings (e.g. "the key moment is no longer inside the clip");
  - approve / reject / undo.

  **Regenerate candidates** re-ranks without repeating extraction or AI. Highlights you approved, rejected or edited are kept and never duplicated.
- The **actual** provider/model used, any fallback, and why are shown on the page and in the job report.

## Phase 2 (long video & audio)

All on the **Long video & audio** page, all local:

- **Audio extraction**: every mapped track becomes a sync-aligned 48 kHz WAV (plus a 16 kHz mono copy for Whisper). The originals are never modified.
- **Three-way mix**: `game_only`, `voice_only`, `both_auto_duck` (FFmpeg `sidechaincompress` + `amix`). Game volume and voice gain are shown in dB; threshold, ratio, attack, release and makeup are adjustable. Modes that cannot work are disabled with the reason. For example, ducking a combined track against the live mic would double your voice.
- **Previews**: an excerpt of every available mode, rendered side by side before you replace the full mix. Previews skip loudness normalisation.
- **Optional quality chain** (all off except loudness/limiter): voice high-pass, FFmpeg `afftdn` noise reduction, two-pass EBU R128 loudness normalisation (default −14 LUFS), peak limiter, fades. The full mix is validated for duration, sample rate and peaks above 0 dBFS.
- **Voiceover studio**: upload WAV/MP3/M4A/FLAC/OGG/OPUS/AAC, or record in the browser (`st.audio_input`). Each version is kept; switch between them. Offset and gain are adjustable, and captions follow the offset.
- **Transcription** (faster-whisper): model `auto`/`tiny`/`base`/`small`/`medium`/`large-v3-turbo`, effective model and device shown, VRAM warning, CUDA-error fallback to CPU. Models download only after you tick the per-model consent box. Writes word timings (`words.json`), `transcript.txt` and `subtitles.srt`. Low-confidence words are flagged. You can edit lines and rebuild subtitles without re-transcribing.
- **Inactive-scene analysis**: a stretch is a cut candidate only when the picture is static **and** game audio is quiet **and** nobody speaks, with pre/post-roll. Sudden game sounds keep it. **Mic silence alone never removes footage.** The default policy removes nothing and only suggests. Every candidate can be kept with a tick. A cut-decision JSON maps source time to output time.
- **Rough cut**: `long_rough_cut.mp4` from the original source, libx264 quality presets (NVENC only after its real encode test), with a stream copy when nothing is cut. The render is validated for duration, video/audio streams, resolution, start/end decodability and non-silent audio. Previous renders are moved to `_previous/`, never overwritten.
- **Timelines**: CMX3600 EDL, FCP7 XML and `timeline.json` (source in/out, track mapping, mix settings). These pass structural self-checks but are **not verified to import into DaVinci Resolve**.
- **Selective re-runs**: changing only the mix re-renders only the mix. Editing the transcript or changing only the voiceover offset rebuilds subtitles, not the transcription. Retry/Resume re-runs the operation that failed, not inspection.

## Phase 1 foundation

- Recursive scan of `01_RAW_INPUT/` (files at the top level or in any game-named sub-folder) and browser upload with size and disk-space checks.
- Free-text game name (no hardcoded game list) with history suggestions; content style; language (Auto / English / Hindi / Hinglish); topic note.
- Duplicate detection (size + sampled fingerprint, then full SHA-256).
- FFprobe inspection: duration, resolution, codec, pixel format, frame rates, rotation, all audio streams (codec, channels, layout, rate, language, title), stream start-time offsets.
- **Measured** variable-frame-rate detection from packet timestamps. Container frame-rate fields alone can miss VFR.
- Decode check at the start and end of the file (catches truncated recordings).
- **Audio track mapping screen**: role suggestions only, a 20-second preview per stream, per-track loudness, and validation. Nothing assumes a fixed OBS layout. Single-track recordings are clearly labelled as inseparable.
- SQLite database with versioned migrations; job state machine (`CREATED → INSPECTING → … → DRAFT_UPLOADED` plus `PAUSED`/`CANCELLED`/`FAILED`); background worker; pause between stages, cancel (stops FFmpeg), retry.
- **Resume**: a stage is reused only when its input fingerprint (source + relevant settings + stage version) matches and its output files still exist. Interrupted stages are detected on restart.
- Output manifests with relative paths, sizes, SHA-256 and validation results, plus `02_WORK/<project>/manifest.json` with settings, software versions and hardware mode (reproducibility).
- Rotating logs (`logs/studio.log`) with secret redaction.
- Truthful hardware detection: each capability is proven by a real test (see below).

## Two-machine workflow

This repository is developed on one PC and run on another:

| Machine | Use |
|---|---|
| Dev PC (no NVIDIA GPU) | Write code, run the automated test suite, commit |
| Run PC (NVIDIA RTX 4050 laptop, 6 GB) | `git clone`, run `setup_windows.bat`, use the app |

Nothing machine-specific is committed: `.env`, `data/` (database, hardware report), `logs/`, media folders, `config/games_history.json` and `.venv/` are git-ignored.

## Installation (Windows 11, 64-bit)

Requirements:

- **Python 3.12** (the only tested version): `winget install Python.Python.3.12`
- **FFmpeg** (full build with libass): `winget install Gyan.FFmpeg`
- Git

Then:

```powershell
git clone <your-repo-url> yt_creator_studio
cd yt_creator_studio
.\setup_windows.bat
```

`setup_windows.bat` checks Windows, Python 3.12 and FFmpeg, creates `.venv`, and installs the pinned `requirements.txt`. If `nvidia-smi` finds a GPU, it asks before installing NVIDIA's CUDA 12 cuBLAS/cuDNN 9 pip packages (a large download, roughly 1 GB or more). It installs optional packages separately, creates `.env` from `.env.example` only if `.env` does not exist, and runs `scripts\verify_installation.py`.

Start the app with `launch_studio.bat`. It verifies the environment and opens the browser.

### Hardware detection

`nvidia-smi` working does **not** mean CUDA inference works, and FFmpeg listing NVENC does **not** mean NVENC works. The app tests each capability separately:

1. `nvidia-smi` device query
2. NVENC listed by FFmpeg, then a **real 1-second NVENC encode**
3. CTranslate2 CUDA device count
4. ONNX Runtime inference (Silero VAD bundled with faster-whisper; no download)
5. faster-whisper CUDA transcription (`--whisper-smoke`, downloads the ~75 MB `tiny` model)

Resulting mode: `CPU`, `NVIDIA_NVENC_ONLY`, `NVIDIA_CUDA_AND_NVENC`, `NVIDIA_CUDA_ONLY` or `GPU_DETECTED_BUT_UNAVAILABLE`. CUDA counts as available only after test 5 passes. The CPU path always remains available.

On the RTX laptop, after setup:

```powershell
.\.venv\Scripts\python.exe scripts\verify_installation.py --whisper-smoke
```

```powershell
.\.venv\Scripts\python.exe scripts\diagnostics.py
```

`diagnostics.py` writes `data\diagnostics_report.json` (no secrets, no personal paths). Share it to confirm GPU support.

**Only one ONNX Runtime is ever installed:** the CPU `onnxruntime` package, which faster-whisper requires. GPU transcription uses CTranslate2's CUDA path, so `onnxruntime-gpu` is not needed. If GPU library installation fails, CPU mode keeps working and no new virtual environment is required.

## OBS recording setup (recommended)

Track numbering differs between OBS setups. The app inspects the actual file and asks you to confirm each track's role. These steps describe a typical setup; menu names may differ slightly between OBS versions.

1. **Settings → Output → Output Mode: Advanced → Recording tab**: tick audio tracks 1, 2 and 3.
2. **Audio Mixer → ⚙ → Advanced Audio Properties**:
   - Desktop/game audio: tracks **1** and **2**
   - Microphone: tracks **1** and **3**
   - Track 1 is then a combined reference mix, track 2 game only, track 3 mic only.
3. Prefer **MKV** recording (survives crashes); MKV is imported directly.
4. **Make a 30-second test recording first**, import it, and check the **Audio track mapping** page. Listen to each 20-second preview before recording a long session.
5. No microphone during play? Record game audio only, then add a post-recorded voiceover on the **Long video & audio** page (Voiceover tab).

If a file has a single audio track, game and microphone are mixed together and **cannot be separated**. The app says so instead of pretending.

## Workflow

1. Copy recordings into `01_RAW_INPUT\` (for example `01_RAW_INPUT\Cricket 24\match1.mkv`) or upload on **Import / New project**.
2. Confirm game name, style and language. **Create project and inspect** starts the background job.
3. **Processing & logs** shows live stage progress, warnings, errors with recovery hints, pause/cancel/retry, and the log.
4. **Audio track mapping**: preview each track, set its role, **Confirm track roles**.
5. **Long video & audio**:
   - **Audio mix**: pick the mode and levels, then **Save & render mix** (the first run extracts the tracks). Use **Render previews of all available modes** to compare.
   - **Voiceover** (optional): upload or record, then re-render the mix and re-transcribe.
   - **Transcript & captions**: choose source and model, allow the model download if asked, then **Save & transcribe**. Correct lines and click **Save edits & rebuild subtitles**.
   - **Cuts & rough cut**: optionally analyse inactive scenes and tick **Keep** where needed, then **Save & render rough cut**. Download EDL/XML or open the output folder.
6. **Analysis & highlights**:
   - **Setup & run**: choose Ollama, Gemini or no AI, and test the connection. For Gemini, read the disclosure and approve cloud analysis for this project. Then **Save & analyse**.
   - **Highlights**: preview, trim, edit the hook/title, approve or reject. **Regenerate candidates only** gives you more or different clips without re-analysing.
7. **Shorts**: adjust the layout and captions of each approved highlight, use **Preview framing**, then **Save & render** (or **Render all**). Watch the result and approve or reject it.
8. **Thumbnails**: **Prepare thumbnail frames & prompts**, copy a prompt into the Gemini web app, and import your finished thumbnail.
9. **Metadata & scripts**: **Generate / update package**, edit any field, then copy fields or download `youtube_pack.txt`.
10. **Settings & diagnostics**: run the capability check and adjust import/inspection settings.

Outputs land in `03_READY_FOR_REVIEW\<project>\` (`long_rough_cut.mp4`, `transcript.txt`, `subtitles.srt`, `words.json`, timelines). Working files are in `02_WORK\<project>\`.

Source recordings are opened read-only and never modified.

## Privacy and cost

- By default everything runs locally. Network access happens only for:
  - an optional, consented Whisper model download from Hugging Face;
  - talking to your local Ollama (`localhost`);
  - Gemini, and only after you approve cloud analysis **for that project**.
- **Gemini consent** (per project, revocable) states what is sent:
  - up to N sampled frames per window at 640 px;
  - the transcript text;
  - timestamps of measured signals;
  - with separate video approval, ≤ 6 low-resolution 360p chunks of the most active windows, deleted after use.

  The full-quality recording is never uploaded. Per Google's pricing page (checked 2026-11-24), Gemini API **free-tier content is used to improve Google's products**.
- Target cost is ₹0. Free-tier availability and limits are set by Google, are not guaranteed, and can change at any time. The app never enables billing or calls a paid endpoint. On a quota error it pauses or falls back to local Ollama. **The app cannot see whether your key's Google Cloud project has billing enabled. Use a key from a project without billing for a hard ₹0 guarantee.**
- Secrets live only in `.env`, which is never committed, logged or stored in the database.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Integration tests generate small synthetic videos with FFmpeg (1/2/3/0 audio tracks, VFR, audio-only, truncated, and a 60 s "gameplay" clip with a static silent loading screen and speech). Speech comes from Windows' built-in offline speech synthesizer. No network, no API calls, no copyrighted media. Phase 2 tests need the Whisper `tiny` model cached locally and are skipped otherwise.

## Tested vs. untested (as of Phase 4)

Phase 4:

| Area | Status |
|---|---|
| Shorts in all 3 layouts (plus split-stack region mode): 1080×1920, exact duration, audio, captions, validation passed | Tested (automated, libx264 `fast`) |
| Crop maths for 16:9, 4:3, 21:9, 9:16 and 4K sources: every crop matches the target shape, so no stretching | Tested (unit) |
| Word-by-word highlight and caption position | Tested, and **checked visually** on rendered frames on the dev PC |
| Only the changed Short re-renders; the mix is reused; title edits never re-render; previous render backed up | Tested (automated) |
| Black-output detection; silent Short for a recording with no audio | Tested (automated) |
| Thumbnail frames (distinct, never upscaled), prompts, user thumbnail import and checks | Tested (automated) |
| Creator package: template (5 titles, placeholders), AI draft validation (repair, invented number flagged, script order, unknown IDs dropped), edits kept on regeneration | Tested (stub provider; no live AI) |
| Shorts, Thumbnails, Metadata pages: render, framing preview, approve, edit and save a title | Tested (Streamlit AppTest) |
| Package written by a **real** AI model (Gemini or Ollama) | **Untested** with a live model. The local 3B model's quality is likely weak (see Phase 3) |
| Hindi / Hinglish package text | **Untested** (template is English only) |
| rembg background cut-outs | **Untested**: rembg isn't installed on the dev PC; the code uses its `remove()` function, so check it on first use |
| NVENC Shorts encoding | **Untested** (needs the RTX laptop) |
| Shorts as they actually look in the YouTube app (UI overlay vs caption margin) | **Not checked**: the overlay area is set by YouTube; the margin is adjustable |
| Fonts other than Arial | **Untested** (libass picks the system font) |

Phase 3:

| Area | Status |
|---|---|
| Signals-only analysis finds the real event in a synthetic recording (loud burst + hard cut + speech) and flags a lone picture cut as weak | Tested (automated) |
| Candidates: bounds, 20–50 s length, speech-safe cuts, handles, de-duplication, kept user decisions on regenerate | Tested (automated) |
| AI validation: unseen frame refs removed, invented quotes removed, out-of-window events dropped, text-only visual claims removed, malformed JSON → one repair → flagged | Tested (fake providers) |
| Gemini: consent required (nothing sent without it), frames mode, video-chunk mode (upload → poll → delete), 429 → Ollama fallback, 429 → pause → resume reusing finished windows | Tested with a **fake** `google-genai` client only |
| **Live Gemini call** (real key, real model list, real quota) | **Untested**: no API key on the dev PC. Run **Test Gemini connection** first |
| Ollama via its HTTP API, including errors (missing model, timeout, length cut-off, text-only refusing images) | Tested (mock transport) |
| Live local Ollama `qwen2.5:3b` (text-only) end to end | Tested on the dev PC: it ran and was validated. Answer quality is weak (see limitations) |
| Local **vision** model through Ollama | **Untested**: none installed on the dev PC |
| Analysis & highlights page: setup, consent buttons, approve, preview | Tested (Streamlit AppTest) |
| Analysis of real multi-hour gameplay: speed, disk use, quality of detected moments | **Not measured yet** (synthetic fixtures only) |

Phase 1–2:

| Area | Status |
|---|---|
| Install via `setup_windows.bat` (CPU path) on Windows 11 + Python 3.12.10 | Tested on dev PC |
| Import, probe, VFR measurement, checksum, track levels, track preview | Tested (automated + manual in browser) |
| Job state machine, resume, pause, cancel, retry, interrupted-stage recovery | Tested (automated) |
| Phase 1 UI pages render; track-role confirmation through the UI | Tested (Streamlit AppTest + manual) |
| Track extraction, mix modes, measured ducking (≥3 dB lower game under speech), mix-only re-run | Tested (automated) |
| Mix previews: excerpt length and voiceover offset sync | Tested (automated) |
| Voiceover upload/versions, offset shifting captions without re-transcribing | Tested (automated) |
| CPU transcription with `tiny`, transcript edits → subtitles without re-transcribing | Tested (automated) |
| Inactive-scene detection (finds only the static silent stretch), keep overrides, no-cut stream copy | Tested (automated) |
| Rough-cut render + validation, previous render backed up | Tested (automated, libx264 `fast` preset) |
| Long video & audio page: renders all tabs, previews, save buttons keep edits/overrides | Tested (Streamlit AppTest) |
| Long video & audio page: visual layout in a real browser | **Not checked yet** |
| Browser microphone recording (`st.audio_input`) | **Untested** (needs a person, a mic and browser permission) |
| Larger Whisper models (`small`…`large-v3-turbo`), Hindi/Hinglish accuracy | **Untested** (only `tiny` + English in tests) |
| EDL / FCP7 XML import into DaVinci Resolve | **Untested**. Structural self-checks only |
| Measured audio/video sync of the rough cut | **Not measured**. Sync relies on cutting audio and video from the same ranges |
| NVIDIA detection, NVENC encode/render, CUDA libraries, CUDA Whisper + CPU fallback | **Implemented, untested on hardware**. Needs the RTX laptop |
| `setup_windows.bat` GPU branch | **Untested** (no NVIDIA GPU on dev PC) |
| Python 3.11 | **Not supported / not tested** |

## Known limitations

- Browser uploads are held in memory by Streamlit (limit 2048 MB). Copy long recordings into `01_RAW_INPUT` instead.
- VFR detection samples the first 30 s by default (configurable).
- A source outside the studio folder is stored with an absolute path and is not portable between PCs.
- The job runner processes one job at a time (by design, to avoid overlapping heavy media work).
- Inactive-scene detection uses only audio level, picture motion and speech. Until Phase 3 adds gameplay analysis, it cannot tell that a quiet, static moment is important. That is why the default policy removes nothing.
- auto-editor integration is **not implemented**. ADR-9 allows it as an optional add-on after a CLI check, but the built-in conservative detector covers the requirement, so nothing calls or installs auto-editor.
- Inline video/audio players are skipped for files over 300 MB (`ui.inline_video_max_mb`), because Streamlit loads them into memory. Use **Open output folder** instead.
- Mix previews skip loudness normalisation, so their absolute level can differ slightly from the final mix.
- **A small local text-only model adds little.** On the dev PC, `qwen2.5:3b` took about 30 s per window on CPU:
  - it placed the event correctly;
  - it over-interpreted it ("a high-scoring moment");
  - it rated no factors;
  - it invented its own categories.

  For real visual understanding, use Gemini (with consent) or install a local vision model yourself.
- Signal detection cannot know game rules. Loud ≠ good, and a scene change may be a menu. Moments without an AI description are capped at "promising" and say so.
- The motion signal uses 160×90 greyscale frames at 2 fps, so small on-screen details (a HUD number changing) are not measured.
- Each analysis frame is a separate exact-seek FFmpeg call. That favours accuracy over speed, but the cost on multi-hour recordings is not measured yet.
- Scores are ranking aids from the evidence available, not objective quality judgements.
- Rendering a Short on the dev PC's CPU (libx264) took about 9–14 s for a 6 s clip, so a 50 s Short takes a few minutes. NVENC on the RTX laptop should be faster, but that's unmeasured.
- The center-crop layout uses about a third of a 16:9 frame's width and upscales it, so sharpness depends on the recording's resolution.
- YouTube limits shown in the app (100-character titles, 5000-character description, ~500 characters of tags, 2 MB / 1280×720 thumbnails) are as known to the app and may change. Verify them in YouTube Studio.
- The number check flags numbers that aren't in the evidence, but it can't catch invented names or claims written without digits. Read the package before publishing.

## Troubleshooting

| Problem | Fix |
|---|---|
| `FFmpeg/FFprobe not found` | `winget install Gyan.FFmpeg`, open a new terminal, or set `FFMPEG_BIN_DIR` in `.env` |
| `existing .venv was created with a different Python version` | Delete `.venv` and rerun `setup_windows.bat` |
| Stage shows `Interrupted` | The app closed mid-stage. Open **Processing & logs** → **Retry**; completed stages are reused |
| GPU present but mode is `GPU_DETECTED_BUT_UNAVAILABLE` | Update the NVIDIA driver, rerun setup and accept the CUDA libraries, then run `verify_installation.py --whisper-smoke` |
| Database "newer than this app supports" | You opened a newer database with older code. Update the code (`git pull`) |
| "Only N GB free… Free up space" | Settings → Disk & temporary files, or move old projects off the drive |
| Gemini "quota or rate limit reached" | The job paused or switched to Ollama; Resume later. Free limits are set by Google |
| Ollama "not reachable" / "model not installed" | Start Ollama; install the model yourself (`ollama pull <name>`), then choose it again |
| Whisper "model is not downloaded yet" | Transcript tab → tick the one-time download box for that model |
| Short failed validation ("black", "silent", size) | Check the layout/crop on Preview framing, re-render that Short; see the stage error on Processing & logs |
| YouTube "Connect again" / 401 | Final approval → Disconnect, then Connect YouTube account |
| YouTube "quota exceeded" | YouTube Data API daily quota; try after it resets. Your approval is kept |

## Roadmap

- ~~**Phase 2**: audio extraction, game/voice/auto-duck mix, voiceover studio, transcription and subtitles, conservative inactivity analysis, rough cut, timeline export~~ (done; Resolve import still unverified).
- ~~**Phase 3**: chunked hybrid analysis, Gemini (opt-in) and Ollama providers, event fusion, explainable scoring, highlight review UI~~ (done; live Gemini and local vision models still unverified).
- ~~**Phase 4**: Shorts in 3 vertical layouts, ASS captions, thumbnail frames and prompts, titles/descriptions/scripts~~ (done; live-AI package text, rembg and NVENC still unverified).
- **Phase 5**: full review dashboard, final approval, optional **private** YouTube upload.
- **Phase 6**: hardening, GPU validation, performance, migration guide.

Design decisions and the dependency validation evidence are in [docs/PHASE0_ADR.md](docs/PHASE0_ADR.md).
