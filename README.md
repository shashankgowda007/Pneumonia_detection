# Pneumonia Detection
# Pneumonia Detection using Vision Transformers and CNNs

A deep learning-powered diagnostic tool that compares **Vision Transformers (ViT)** and **Convolutional Neural Networks (CNN)** for **automated pneumonia detection** on chest X-ray images. This project is implemented with **Streamlit** to provide a real-time web-based diagnostic interface for clinical use.

---

## 🔍 Project Overview

Pneumonia is a leading cause of death globally, yet its diagnosis from chest X-rays remains error-prone in emergency departments. This project compares the diagnostic performance and interpretability of Vision Transformers (ViTs) and CNNs, providing a real-time, explainable AI interface using attention visualizations.

---

## 🧠 Key Features

- 🧪 Comparative analysis of CNN (DenseNet121) vs Vision Transformer
- 📊 Attention map visualization for clinical interpretability
- 🖼️ Support for JPEG and DICOM image formats
- ⏱️ Fast inference (<15s per image)
- 🧑‍⚕️ Usable by clinicians via Streamlit interface
- 📈 Evaluation based on NIH ChestX-ray14 dataset

---

## 📁 Dataset

**NIH ChestX-ray14 Dataset** (112,120 images from 30,805 patients)

📦 Download link: [NIH ChestX-ray Dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC)

**Instructions:**

1. Download and extract the dataset from the link above.
2. Create the following folders inside your project:
    ```
    data/
      ├── Normal/
      └── Pneumonia/
    ```
3. Move corresponding chest X-ray images to the appropriate folder based on their label (Normal or Pneumonia).

---

## 📦 Installation

Create a virtual environment and install the required libraries:

```bash
pip install -r requirements.txt


```Run the code 

streamlit run app.py
