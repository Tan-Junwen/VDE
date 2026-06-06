# ⚡ Qwen-Image Inference Acceleration via VDE
[**VDE**](https://github.com/YourOrg/VDE) naturally supports accelerated inference for[**Qwen-Image**](https://github.com/QwenLM/Qwen-Image) by shifting the paradigm from caching-and-reusing to decomposing-and-estimating. We provide two optional acceleration paths based on the balance between quality and speed.

---

## 📊 Inference Latency 

**Comparisons on a Single NVIDIA A800 @ 512×512:**

| Method              | Qwen-Image (T=50) | VDE-slow | VDE-fast |
|:-------------------:|:-----------------:|:--------:|:--------:|
| **Latency**         | 12.53 s           | 6.14 s   | 4.64 s   |
| **Speedup**         | 1.00×             | 2.04×    | 2.70×    |
| **T2I**             | <img width="160" alt="Qwen-Image" src="[YOUR_IMAGE_LINK_HERE]" /> | <img width="160" alt="VDE-slow" src="[YOUR_IMAGE_LINK_HERE]" /> | <img width="160" alt="VDE-fast" src="[YOUR_IMAGE_LINK_HERE]" /> |

> 💡 *Note: VDE completely estimates the velocity output online directly from the current input, avoiding cache-input miamatch seen in traditional caching methods.*

---

## 🛠️ Installation

Please refer to [Qwen-Image](https://github.com/QwenLM/Qwen-Image) for base environment setup.

## Usage

VDE provides two acceleration settings controlled by `--stable-step` and `--interval`.

```bash
# vanilla Qwen-Image (no acceleration)
python inference_qwenimage.py

# VDE acceleration settings
python inference_qwenimage.py --vde --stable-step 10 --interval 3   # slow
python inference_qwenimage.py --vde --stable-step 10 --interval 5   # fast
```

## 📖 Citation
If you find **VDE** useful in your research or applications, please consider giving us a star ⭐ and citing it by the following BibTeX entry:

```bibtex
@inproceedings{tan2026vde,
  title={VDE: Training-Free Accelerating Rectified Flow Model via Velocity Decomposition and Estimation},
  author={Tan, Junwen and Liang, Jinglin and Chen, Hongyuan and Huang, Shuangping},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={37918--37928},
  year={2026}
}
```

## Acknowledgements

We would like to thank the contributors to the  [Qwen-Image](https://github.com/QwenLM/Qwen-Image), and [Diffusers](https://github.com/huggingface/diffusers).
