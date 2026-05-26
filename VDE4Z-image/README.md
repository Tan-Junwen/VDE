# VDE Inference Examples

This repository contains single-file inference examples for VDE acceleration on FLUX.1, Qwen-Image, Wan2.1, and Z-Image.

The working tree keeps two kinds of files:

- Local runnable files in `VDE4*/`, which may use local model and diffusers paths through environment-variable defaults.
- GitHub release files generated under `github/`, which remove private local paths and use public model identifiers or explicit CLI arguments.

See `LOCAL_RUNS.md` for the local path checklist.

## Local Smoke Test

Run syntax checks first:

```bash
python scripts/check_python_syntax.py VDE4FLUX/inference_flux1.py VDE4QwenImage/inference_qwenimage.py VDE4Wan2.1/inference_wan.py VDE4Z-Image/inference_z-image.py VDE4Z-Image/inference_z-image-turbo.py
```

Then run a model-specific script after setting the required local paths:

```bash
python scripts/check_runtime_env.py --target flux
bash VDE4FLUX/run_inference_flux1.sh
python scripts/check_runtime_env.py --target qwenimage
bash VDE4QwenImage/run_inference_qwenimage.sh
python scripts/check_runtime_env.py --target wan
bash VDE4Wan2.1/run_inference_wan.sh
python scripts/check_runtime_env.py --target zimage
bash VDE4Z-Image/run_inference_z-image.sh
python scripts/check_runtime_env.py --target zimage-turbo
bash VDE4Z-Image/run_inference_z-image-turbo.sh
```

## Prepare GitHub Files

From PowerShell:

```powershell
.\scripts\prepare_github_release.ps1
```

The generated `github/` directory is the version intended for publication. Before pushing, inspect it with:

```bash
rg -n "/(fsave)|t[j]w|c[h]y|C:\\\\U[s]ers|[YX]:" github
python scripts/check_python_syntax.py github/VDE4FLUX/inference_flux1.py github/VDE4QwenImage/inference_qwenimage.py github/VDE4Wan2.1/inference_wan.py github/VDE4Z-Image/inference_z-image.py github/VDE4Z-Image/inference_z-image-turbo.py
```
