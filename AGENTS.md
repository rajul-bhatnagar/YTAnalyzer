# AGENTS.md: YT Creator Studio

Local-first Windows 11 app that turns a long gameplay recording into a review-gated YouTube package: a long video, Shorts, thumbnails, a script, captions and metadata. Read `README.md` for user setup. This file is for agents working on the code.

## Hard rules (do not break)
- **₹0 only.** Use Gemini free tier, Ollama and local tools. Never call a paid API, enable billing or add a paid dependency.
- **Human approval gates stay.** Nothing is auto-published. YouTube upload is private by default and needs explicit confirmation plus final approval (`publish_service.final_check` / `approve_final`).
- **Check AI output against evidence.** AI JSON is validated against the measured signals, the transcript and OCR. Invented numbers are flagged (number guard, `short_factcheck`). Never present AI guesses as facts.
- **Game-agnostic.** No hardcoded game names or catalogues. Per-genre behaviour lives only in `pipeline/style_profiles.py`.
- **Backward-compatible, resumable state.** DB changes go in as new versioned migrations in `pipeline/database.py` (currently schema v4). Stage fingerprints decide reuse: bump a fingerprint version when output semantics change.
- **Platform.** Windows 11, Python 3.11/3.12 (3.12.10 tested). CPU must always work. NVIDIA (RTX 4050 6 GB) is an optional path, enabled only after a real CUDA/NVENC test.
- **Secrets.** Never commit `.env`, API keys, OAuth tokens, client secrets or generated media. Never ask the user to paste keys in chat. Test keys in tests are built from pieces to avoid secret scanners.
- **No silent downloads.** Never install models or packages without asking. `rapidocr` and Ollama models are optional and installed by the user.
- **Git.** Commit as `Rajul Bhatnagar <118436631+rajul-bhatnagar@users.noreply.github.com>` with **no** Claude `Co-Authored-By` trailer. Ask before pushing. Remote: `github.com/rajul-bhatnagar/YTAnalyzer`, branch `main`.

## Architecture
```
app.py (Streamlit 1.67)  ->  ui/*.py pages  ->  StudioService (pipeline/project_service.py)
                                                 ├─ AnalysisServiceMixin  (analysis_service.py)
                                                 ├─ ContentServiceMixin   (content_service.py)
                                                 └─ PublishServiceMixin   (publish_service.py)
StudioService -> JobRunner (job_manager.py, single worker thread) -> StageSpec lists
              -> SQLite (database.py) + per-project files (paths.py / longform.ProjectFiles)
```
- **Job states:** CREATED → INSPECTING → AUDIO_PREP → ANALYZING → HIGHLIGHT_REVIEW → EDITING → SHORTS_RENDER → ASSET_PREP → FINAL_REVIEW → APPROVED → DRAFT_UPLOADED. PAUSED, CANCELLED and FAILED are side states. A stage raises `StagePaused` to wait for the user.
- **Stages:** each `StageSpec` has inputs → fingerprint → exec. An unchanged fingerprint reuses the previous output. Stage lists live in `longform.build_stages` (inspection, audio, transcription, mix, rough cut), `analysis_stages.ANALYSIS_STAGES` (`signal_events, frames, hud_ocr, ai_windows, highlights`) and `content_stages` (Shorts, thumbnails, creator pack). The service maps targets to stages (`CONTENT_TARGETS`, including `voiceover`).
- **Analysis:** signal events (audio, scene and activity) plus optional HUD OCR feed AI windows (30 s overlap). Windows are ranked and gated by `ai.max_ai_windows` plus `ai.pinned_ranges`. Results go through validation and cross-window merge into scored highlight candidates with `outcome_mood`.
- **AI providers** (`pipeline/ai_providers/`): Gemini (google-genai, JSON schema), with Ollama as a local fallback. `ProviderPlan`/`ProviderSlot` are in `gameplay_analysis.py`. Results are cached per window. Retries use `ai.retry_delays_sec` (30/60/120) for cloud and 5/20 s for local. Before CUDA Whisper, the Ollama model is unloaded and VRAM freed.
- **Media:** all FFmpeg calls go through `ffmpeg_tools.py`. Filter graphs are passed as files. Captions use a cwd-relative `ass=captions.ass`. Audio is ducked with sidechaincompress and normalised with two-pass loudnorm.
  - **A/V sync:** the seek shift is `media_probe.video_seek_shift(info)` = video start − earliest stream start. Apply it to every trim or seek on the source video (timeline, Shorts, previews).
  - **FFmpeg 9.1 AAC encoder stalls on mono input.** Always force `-ac 2` on AAC output.
- **Shorts:** `shorts_builder.py` renders the three layouts (`blur_fit`, `center_crop`, `split_stack` with a free-shape bottom region). `short_edit.py` plans dead-air trim or speed-up and remaps caption words. `short_factcheck.py` checks the hook and title against evidence. Validation warns rather than fails when the source itself is silent.

## Folder and file conventions
- `pipeline/` holds pure logic and services, with no Streamlit imports. `ui/` holds one module per page with a `render()` function; shared widgets go in `ui/common.py`.
- `config/default_settings.json` holds the defaults with dotted keys (`shorts.layout`, `ai.provider` and so on). Per-project overrides use `save_project_settings`. An empty or null value means "use the style profile".
- `01_RAW_INPUT/`, `02_WORK/` and `03_READY_FOR_REVIEW/` are user data roots (ignored by git, as are `data/` and `logs/`). Generated files are versioned with `_previous/` backups.
- **Tests:** `tests/unit/` and `tests/integration/`. Integration tests use a synthetic 90 s recording (`conftest.media`), need FFmpeg and the Whisper `tiny` model, and use fake Gemini or YouTube clients (never the network). Fixtures that set `analysis.window_sec: 45` must also pin `analysis.window_overlap_sec: 0`. UI tests use Streamlit `AppTest`.
- **Commands:**
  - Tests: `.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider`. It is slow, so run it in the background with the full output saved.
  - Machine check: `.venv\Scripts\python.exe scripts\diagnostics.py`.
- **Two machines:** this dev PC writes and tests code (CPU only). The app runs on an ASUS ROG G16 with an RTX 4050 via `git pull`. The laptop must not be in Armoury Crate *Eco* mode, which turns off the GPU.

## Current status (2026-11-25)
- Phases 0–6 are complete and pushed. The latest commit, `7dcfc97`, adds:
  - style profiles
  - long-recording AI gating and window overlap
  - per-Short fact-check and dead-air trim or speed-up
  - free-shape split crop and a safe-zone overlay
  - a voiceover quick loop (whole video and per Short)
  - optional rapidocr HUD OCR
  - `outcome_mood` and a Pillow thumbnail text composer
  - the VFR A/V seek fix, `file://localhost` EDL URIs and YouTube quota/OAuth notices
- The RTX laptop is verified as NVIDIA_CUDA_AND_NVENC, and Gemini works on a real 17.5 min mobile cricket recording.

## Known issues and open risks
- **The full pytest suite has NOT been run on `7dcfc97`**; only targeted tests ran. Run it first.
- The `rapidocr` integration (`hud_ocr._engine`) has not been exercised against the real package.
- Gemini free tier often returns 503 on some windows. Re-running analysis retries the failed windows (retry token), but results can be partial.
- Highlight cards can repeat the same measurement line. A dedup fix was deferred by the user.
- The user's recordings may have a mono, mostly silent track. The pipeline must keep handling silent sources (warn, don't fail).

## Next steps
1. Run the full test suite on `main` and fix any regressions. Ask before pushing.
2. Run these checks on the RTX laptop:
   - re-render the Super Over Short (sync, Sports split-strip default)
   - check the safe-zone overlay
   - check AI gating on a long recording
   - check local-vision VRAM unload (if Ollama vision is installed)
   - check the thumbnail composer
3. If the user installs rapidocr, verify HUD OCR on real frames and tune `analysis.hud_bands`.
4. Optional: dedup repeated evidence lines in highlight cards (only if the user asks).
