# ⚡ FLUX Inference Acceleration via VDE

[**FLUX.1**](https://github.com/black-forest-labs/flux) and [**FLUX.2**](https://github.com/black-forest-labs/flux2) are high-performance text-to-image and image-to-image diffusion frameworks built by *Black Forest Labs*. [**VDE**](https://github.com/YourOrg/VDE) (**Velocity Decomposition and Estimation**) now supports FLUX, providing multiple training-free acceleration modes that balance **quality vs. speed** 🚀 by fundamentally resolving the cache-input mismatch problem.

---

## 📊 Inference Latency

**Example latency comparisons on a single NVIDIA A100 @ 1024×1024:**

| Method              | FLUX.1 (T=50) | VDE-slow (Ours) | VDE-medium (Ours) | VDE-fast (Ours) |
|:-------------------:|:-------------:|:---------------:|:-----------------:|:---------------:|
| **Latency**         | 8.20 s        | 3.70 s          | 3.04 s            | 2.72 s          |
| **Speedup**         | 1.00×         | 2.21×           | 2.70×             | 3.01×           |
| **SSIM ↑**          | -             | 0.8877          | 0.8499            | 0.8267          |
| **T2I**             | <img width="120" alt="Flux Original" src="[YOUR_IMAGE_LINK_HERE]" /> | <img width="120" alt="VDE-slow" src="[YOUR_IMAGE_LINK_HERE]" /> | <img width="120" alt="VDE-medium" src="[YOUR_IMAGE_LINK_HERE]" /> | <img width="120" alt="VDE-fast" src="[YOUR_IMAGE_LINK_HERE]" /> |

> 💡 Numbers above are example measurements; actual latency may vary depending on resolution, batch size, and hardware configuration. VDE consistently preserves structural integrity and fine details much better than naive step reduction or feature caching.

---

## 🛠️ Installation & Usage

Please refer to the official projects for base installation instructions:
-[**FLUX.1**](https://github.com/black-forest-labs/flux)

## Usage

VDE provides several acceleration settings controlled by `--stable-step` and `--interval`.

```bash
# vanilla FLUX (no acceleration)
python inference_flux1.py

# VDE acceleration settings
python inference_flux1.py --vde --stable-step 6 --interval 3   # slow
python inference_flux1.py --vde --stable-step 6 --interval 4   # medium
python inference_flux1.py --vde --stable-step 6 --interval 5   # fast

## 📖 Citation
If you find **VDE** useful in your research or applications, please consider giving us a star ⭐ and citing it by the following BibTeX entry:

```bibtex
@inproceedings{tan2026vde,
  title     = {VDE: Training-Free Accelerating Rectified Flow Model via Velocity Decomposition and Estimation},
  author    = {Junwen Tan and Jinglin Liang and Hongyuan Chen and Shuangping Huang},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

## Acknowledgements

We would like to thank the contributors to the  [FLUX](https://github.com/black-forest-labs/flux), and [Diffusers](https://github.com/huggingface/diffusers).
