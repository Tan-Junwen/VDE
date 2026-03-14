# ⚡ Wan2.1 Inference Acceleration via VDE
[**VDE**](https://github.com/YourOrg/VDE) natively extends to the temporal domain, providing seamless training-free acceleration for **[Wan2.1](https://github.com/Wan-Video/Wan2.1)**. By decomposing velocity trajectories, VDE maintains highly consistent spatiotemporal features while dramatically reducing video generation time.

---

## 📊 Inference Latency

**Latency comparisons on Wan2.1-1.3B (81 frames, 832×480) on a single NVIDIA A800:**

| Method              | Wan2.1 (T=50) | VDE-slow | VDE-fast |
|:-------------------:|:-------------:|:--------:|:--------:|
| **Latency**         | 175.35 s      | 84.18 s  | 70.11 s  |
| **Speedup**         | 1.00×         | 2.08×    | 2.50×    |
| **VBench Score**    | 81.30%        | 80.32%   | 80.43%   |
| **T2V (GIF)**       | <img width="160" alt="Original" src="[YOUR_GIF_LINK_HERE]" /> | <img width="160" alt="VDE-slow" src="[YOUR_GIF_LINK_HERE]" /> | <img width="160" alt="VDE-fast" src="[YOUR_GIF_LINK_HERE]" /> |

> 💡 *Note: VDE preserves dynamic motion and subject consistency significantly better than naive step reduction or traditional feature caching.*

---

## 🛠️ Installation & Usage

Please refer to the official [Wan2.1](https://github.com/Wan-Video/Wan2.1) repository for environment setup.

### Usage 

```bash
# vanilla Wan2.1 (no acceleration)
python inference_wan21.py

# VDE acceleration modes
python inference_wan21.py --vde_mode slow
python inference_wan21.py --vde_mode fast
