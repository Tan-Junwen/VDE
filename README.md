<div align="center">
  <img src="./assets/logo.png" style="width:auto; height:120px;" alt="VDE Logo">
</div>

# [CVPR 2026] VDE: Training-Free Accelerating Rectified Flow Model via Velocity Decomposition and Estimation

<div align="center" style="font-size: 20px; font-weight: 700;">
  <a href="https://github.com/Tan-Junwen" target="_blank">Junwen Tan</a>,
  <span class="author-block">Jinglin Liang</span>,
  <span class="author-block">Hongyuan Chen</span>,
  <span class="author-block">Shuangping Huang</span>
</div>

<div align="center" style="font-size: 18px; font-weight: 500; margin-top: 6px;">
  <span class="author-block">South China University of Technology</span>
</div>

<h5 align="center">

<a href="https://github.com/Tan-Junwen/VDE" target="_blank">
  <img src="https://img.shields.io/badge/Project-Website-blue.svg" alt="Project Page">
</a>
<a href="https://arxiv.org/pdf/2605.23381" target="_blank">
  <img src="https://img.shields.io/badge/Paper-PDF-critical.svg?logo=adobeacrobatreader" alt="Paper">
</a>
<a href="./LICENSE" target="_blank">
  <img src="https://img.shields.io/badge/License-Apache%202.0-yellow.svg" alt="License">
</a>
<a href="https://github.com/Tan-Junwen/VDE/stargazers" target="_blank">
  <img src="https://img.shields.io/github/stars/Tan-Junwen/VDE.svg?style=social" alt="GitHub Stars">
</a>

</h5>

![VDE Overview](./assets/cover_01.png)
> **Figure 1.** Comparison between VDE and standard 50-step sampling across Flux, Qwen-Image, and Wan2.1. VDE achieves comparable visual quality with dramatically reduced runtime (up to 3.01× speedup).

## 💡 Introduction

Though Rectified Flow (RF) models have achieved remarkable performance in visual generation, their practical deployments are challenged by slow inference speeds. Previous training-free acceleration methods typically follow a **caching-and-reusing** paradigm, neglecting the growing mismatch between static cached values and evolving inputs.

We propose **Velocity Decomposition and Estimation (VDE)**, a novel method that shifts the paradigm from *caching-and-reusing* to **decomposing-and-estimating**. 
- VDE decomposes the model's velocity output into components parallel and orthogonal to the input.
- It exploits the temporal predictability of the components' coefficients and the consistency of the orthogonal direction.
- VDE periodically anchors the model's state and precisely estimates subsequent outputs analytically in an inherently **input-adaptive** manner.

VDE achieves up to **2.04× - 3.22× acceleration** with minimal loss in visual quality, outperforming the best cache-based baseline by **19.5% in SSIM**, **30.3% in PSNR**, and reducing **LPIPS by 55.4%** in image generation.

---

## 🔥 Latest News
<!--
- [2026/03/xx] ✨ **ComfyUI-VDE** is now available! Enjoy VDE acceleration directly in your ComfyUI workflows.
-->
- [2026/05/30] 🚀 The Code for **VDE** is officially released! Support Image and Video generation/editing.
- [2026/05/22] 📄 Our paper is available on [arXiv](https://arxiv.org/pdf/2605.23381).
- [2026/02/21] 🎉 **VDE** is accepted by **CVPR 2026**! 

---

## 🛠️ Supported Models

VDE is highly versatile and supports a wide range of state-of-the-art Rectified Flow models across modalities:

🎨 **Image Generation**
- [x] [FLUX.1-dev](https://github.com/black-forest-labs/flux)
- [x] [Qwen-Image](https://github.com/QwenLM/Qwen-Image)
- [x] [Z-Image](https://github.com/Tongyi-MAI/Z-Image)
- [x] [HiDream](https://github.com/HiDream-ai/HiDream-I1)

🎥 **Video Generation**
- [x][Wan2.1](https://github.com/Wan-Video/Wan2.1) (T2V & I2V)
- [x] [HunyuanVideo](https://github.com/Tencent/HunyuanVideo)
- [x] [Open-Sora](https://github.com/hpcaitech/Open-Sora)
- [x] [Open-Sora-Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan)

🧊 **3D Generation**
- [x] [Trellis2](https://github.com/microsoft/TRELLIS)

<!--
---
## 🧩 Ecosystem & Integrations

We are actively working to integrate VDE into the broader open-source generative AI ecosystem. If you are a developer, we welcome PRs!

- **[ComfyUI-VDE]([Link_to_your_ComfyUI_Repo])**: We provide a custom node for **ComfyUI**, allowing regular users to achieve 2x~3x faster generation in their favorite workflows without writing code.
- **[Cache-DiT](https://github.com/vipshop/cache-dit)** & **[x-DiT](https://github.com/xdit-project/xDiT)**: VDE is compatible with multi-GPU and system-level acceleration engines. *(Integration in progress)*
- **[Diffusers](https://github.com/huggingface/diffusers)**: We are preparing upstream PRs to bring VDE natively into Hugging Face `diffusers`.

---
-->

## ⚡ Performance & Demos

### 1. FLUX-dev (Text-to-Image)
**Baseline Latency (T=50): 8.20s**
| Method | Speedup | Latency | SSIM (↑) | PSNR (↑) | LPIPS (↓) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **VDE-slow** | **2.21×** | 3.70 s | **0.8877** | **25.81** | **0.1243** |
| *EasyCache-fast* | *2.91×* | *2.81 s* | *0.7240* | *19.59* | *0.3197* |

### 2. Qwen-Image (Text-to-Image)
**Baseline Latency (T=50): 12.53s**
| Method | Speedup | Latency | SSIM (↑) | PSNR (↑) | LPIPS (↓) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **VDE-slow** | **2.04×** | 6.14 s | **0.9362** | **28.58** | **0.0691** |
| *TeaCache-fast* | *2.69×* | *4.66 s* | *0.5596* | *14.43* | *0.4773* |

### 3. Wan2.1 (Text-to-Video)
**Baseline Latency (T=50, 81 frames, 832×480): 175.35s**
| Method | Speedup | Latency | VBench (%) ↑ | SSIM (↑) | LPIPS (↓) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **VDE-slow** | **2.08×** | 84.18 s | **80.32** | **0.8902** | **0.0554** |

---

## 🚀 Getting Started

### Installation
```bash
git clone https://github.com/[Your_GitHub_Username]/VDE.git
cd VDE
conda create -n vde python=3.10
conda activate vde
pip install -r requirements.txt
```

### Inference Example (FLUX-dev)
```python
from vde.pipeline import VDEFluxPipeline
import torch

pipe = VDEFluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.to("cuda")

# Using VDE-slow setting (2.21x speedup, highest quality)
image = pipe(
    prompt="A slice of cake with frosting on a plate",
    num_inference_steps=50,
    vde_mode="slow" 
).images[0]

image.save("output_vde.jpg")
```
*For detailed scripts on other models (Wan2.1, Qwen-Image, etc.), please refer to the `examples/` folder.*

---

## 📋 To-Do List
- [x] Release core VDE algorithm and Paper.
- [x] Support Text-to-Image (FLUX, Qwen, Z-Image, HiDream).
- [x] Support Text-to-Video (Wan2.1, HunyuanVideo, Open-Sora).
- [ ] Release ComfyUI Custom Nodes.
- [ ] Upstream PR to Hugging Face `diffusers`.

---

## 💐 Acknowledgement

This project builds upon several excellent open-source projects, including [Diffusers](https://github.com/huggingface/diffusers), [FLUX](https://github.com/black-forest-labs/flux), [Qwen-Image](https://github.com/QwenLM/Qwen-Image), [Z-Image](https://github.com/Tongyi-MAI/Z-Image), [Wan2.1](https://github.com/Wan-Video/Wan2.1), and [HunyuanVideo](https://github.com/Tencent/HunyuanVideo). We sincerely thank the authors for their contributions to the community.

---

## 🔒 License
This project is licensed under the [Apache License 2.0](LICENSE). 

---

## 📖 Citation
If you find VDE useful for your research or applications, please consider giving us a star ⭐ and citing our paper:

```bibtex
@inproceedings{vde2026,
  title     = {VDE: Training-Free Accelerating Rectified Flow Model via Velocity Decomposition and Estimation},
  author    = {[Your Name 1] and [Your Name 2] and [Your Name 3]},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

<div align='center'>
<a href="https://star-history.com/#[Your_GitHub_Username]/VDE&Date">
  <picture align='center'>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=[Your_GitHub_Username]/VDE&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=[Your_GitHub_Username]/VDE&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=[Your_GitHub_Username]/VDE&type=Date" width=400px />
  </picture>
</a>
</div>
