# 📄 PixelPrompt: Parameter-Efficient NR-IQA via Pixel-Space Visual Prompts in Multimodal LLMs

![teaser-image](./assets/teaser.png)  

---

## ✍️ Authors

- [Yahya Benmahane](https://www.linkedin.com/in/yahya-benmahane/) — Computer Science Departement | Faculty of Sciences, Rabat
- [Mohammed El Hassouni](https://scholar.google.com/citations?user=aIwj9L0AAAAJ&hl=fr) — Professor, Computer Science, Mohammed V University in Rabat, Morocco  

---

## 📜 Abstract

> In this paper, we propose PixelPrompt, a novel parameter-efficient adaptation method for No-Reference Image Quality Assessment (NR-IQA) using visual prompts optimized in pixel-space. Unlike full fine-tuning of Multimodal Large Language Models (MLLMs), our approach optimizes a negligible number of learnable parameters while keeping the base MLLM entirely fixed. During inference, these visual prompts are combined with images via addition and processed by the MLLM with the textual query "Rate the technical quality of the image." Extensive evaluations across distortion types (synthetic, realistic, AI-generated) on KADID-10k, KonIQ-10k, and AGIQA-3k demonstrate competitive performance against full finetuned methods, achieving 0.91 SROCC on KADID-10k. To our knowledge, this is the first work to leverage pixel-space visual prompts for NR-IQA, enabling efficient MLLM adaptation for low-level vision tasks.

---

## 📌 Citation

```bibtex
@article{your2025paper,
  title={Paper Title},
  author={Your Name and Coauthor Name and Another Author},
  journal={Conference/Journal Name},
  year={2025}
}
