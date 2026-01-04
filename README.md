# Human Face Emotion Classification using Vision Transformer (ViT)

This project demonstrates how to fine-tune a pretrained **Vision Transformer (ViT)** model from Hugging Face for **human face emotion classification** using a Kaggle image dataset.  
The implementation is done in **PyTorch** with **Hugging Face Transformers** and was originally developed in **Google Colab**.

---

## 📌 Project Overview

- **Task**: Image classification (Human facial emotions)
- **Dataset**: Kaggle – Human Face Emotions
- **Model**: `Falconsai/nsfw_image_detection` (ViT-based image classification model)
- **Frameworks**:
  - PyTorch
  - Hugging Face Transformers
  - Torchvision
- **Training Strategy**:
  - Freeze backbone (ViT)
  - Train only the classifier head
  - Stratified split (per-class train/val/test)

---

## 🗂 Dataset

The dataset is automatically downloaded from Kaggle using `kagglehub`.

```python
kagglehub.dataset_download("samithsachidanandan/human-face-emotions")
