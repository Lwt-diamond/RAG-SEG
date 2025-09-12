
# RAG-SEG: First RAG, Second SEG
[![Paper](https://img.shields.io/badge/arXiv-2508.15313-b31b1b.svg)](https://arxiv.org/abs/2508.15313)
[![Demo](https://img.shields.io/badge/🤗-Demo%20Space-blue.svg)](https://huggingface.co/spaces/Sherry4869/RAG-SEG)

> **First RAG, Second SEG: A Training-Free Paradigm for Camouflaged Object Detection**  
> RAG-SEG leverages Retrieval-Augmented Generation (RAG) to produce pseudo prompts, followed by SAM-based segmentation.  
> On **camouflaged object detection (COD)**, RAG-SEG achieves **SOTA performance among training-free methods**, and on **salient object detection (SOD)**, it improves **inference speed**.

---

## 🚀 Highlights
- **Fast inference**: Lower GPU memory usage and faster inference than existing training-free COD methods.  
- **Generalizable**: Directly applicable to **SOD** and potentially extendable to other datasets.  

---

## ⚙️ Installation
```bash
# Create environment
conda create -n py310 python==3.10
conda activate py310

# Install dependencies (using uv for speed)
pip install uv
uv pip install -r requirements.txt

# Run demo
python app.py
````

---

## 📊 Visualization

**Segmentation results:**

<p align="center">
  <img src="./demo_images/all_seg.png" width="700">
</p>

**Demo running example:**

<p align="center">
  <img src="./demo_images/Snipaste_2025-09-10_15-18-34.png" width="700">
</p>

For online demo, visit: [RAG-SEG HuggingFace Space](https://huggingface.co/spaces/Sherry4869/RAG-SEG)

⚠️ 建议在 本地运行 本项目，因为在 HuggingFace Space 上由于网络和 CPU 限制，推理速度可能较慢。
---

## 📖 Citation

If you find this work useful, please cite:

```bibtex
@article{liu2025first,
  title={First RAG, Second SEG: A Training-Free Paradigm for Camouflaged Object Detection},
  author={Liu, Wutao and Wang, YiDan and Gao, Pan},
  journal={arXiv preprint arXiv:2508.15313},
  year={2025}
}
```


## 📌 License

This repository is released under the MIT License.

