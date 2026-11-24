# Moving from the dev PC to the RTX 4050 laptop

The repository is portable: no absolute paths, and machine data (`.env`, `data/`, `logs/`, media folders, `.venv/`) is git-ignored. GPU support is decided **on the run laptop by real tests**, never assumed.

## 1. On the dev PC

1. Commit and push the repository to GitHub (private repository recommended):

   ```powershell
   git add -A
   ```

   ```powershell
   git commit -m "YT Creator Studio"
   ```

   ```powershell
   git push
   ```

2. Media, projects and the database stay on the dev PC. To move a project, copy its source recording into `01_RAW_INPUT` on the laptop and import it again. The project database is not meant to be copied between PCs.

## 2. On the RTX 4050 laptop

1. Install **Python 3.12** (`winget install Python.Python.3.12`), **FFmpeg** (`winget install Gyan.FFmpeg`), Git, and a current **NVIDIA driver**.
2. Clone the repository and run setup. When it asks, accept the NVIDIA CUDA 12 cuBLAS/cuDNN 9 libraries (a large download, roughly 1 GB or more):

   ```powershell
   git clone <your-repo-url> yt_creator_studio
   ```

   ```powershell
   cd yt_creator_studio; .\setup_windows.bat
   ```

3. Prove the GPU paths. Each one is used only after its own test passes:

   ```powershell
   .\.venv\Scripts\python.exe scripts\verify_installation.py --whisper-smoke
   ```

   ```powershell
   .\.venv\Scripts\python.exe scripts\diagnostics.py
   ```

   `data\diagnostics_report.json` should report `NVIDIA_CUDA_AND_NVENC`. If it reports `GPU_DETECTED_BUT_UNAVAILABLE`, see Troubleshooting in the README. The CPU path keeps working either way.
4. Copy `.env.example` → `.env` if setup did not, then fill in only what you use: `GEMINI_API_KEY`, `OLLAMA_MODEL`, `YOUTUBE_CLIENT_SECRETS_FILE` (a path to the OAuth JSON kept **outside** the repository).
5. Start with `launch_studio.bat`, open **Settings & diagnostics** and run the capability check.

## 3. First real checks (not possible on the dev PC)

| Check | How |
|---|---|
| NVENC render | Render a rough cut and a Short with encoder "auto"; the job report shows `h264_nvenc` |
| CUDA Whisper | Transcribe with model `auto`; the transcript header shows device `cuda` and the effective model |
| Gemini | Analysis & highlights → Test Gemini connection, then approve consent for one short test project |
| Local vision model | Install one yourself in Ollama; the model list must show "vision" |
| YouTube | Final approval → Connect YouTube account → upload a test Short as **private** |
| DaVinci Resolve | Import `long_rough_cut_fcp7.xml` or `.edl`, and note your Resolve version |

The 6 GB of VRAM is shared: Ollama models and large Whisper models can't run side by side. The app warns before choosing a Whisper model that may not fit, and falls back to CPU on a CUDA out-of-memory error.
