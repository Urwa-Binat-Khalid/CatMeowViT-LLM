# CatMeowViT-LLM
# Cat Meow Translator with Vision Transformer and LLM Explanation

This project is an AI-powered system that classifies cat vocalizations from audio recordings and provides detailed behavioral explanations using a large language model (LLM). The model predicts the likely meaning behind your cat’s sounds and advises you on how to respond.

---

##  Overview

- **Audio Classification:** Uses a Vision Transformer (ViT) model pretrained on Mel spectrogram images derived from cat sounds to classify vocalizations into categories like "Brushing", "WaitingForFood", and "Isolation".
- **Large Language Model Explanation:** Once the sound is classified, a Groq-hosted LLM (based on LLaMA 3) generates a detailed explanation of the cat’s behavior and owner guidance.
- **Interactive Gradio UI:** Users can upload their cat’s meow audio and receive instant classification and behavioral interpretation in a user-friendly web interface.
---

##  Features

- Audio preprocessing pipeline converting raw `.wav` files into normalized 3-channel Mel spectrogram images.
- Dataset balancing and augmentation to improve model robustness.
- Fine-tuned ViT model for sound classification.
- Integration with Groq LLM API for natural language explanations.
- Gradio app for easy audio upload, prediction, and interactive explanation.

---

## Getting Started

### Requirements

- Python 3.8+
- GPU recommended for faster training and inference
- Libraries:
  - torch, torchvision, torchaudio
  - timm
  - pandas, numpy
  - scikit-learn
  - matplotlib
  - gradio

### Code Overview
data.py: Loads .wav files, assigns labels, balances classes, and outputs a CSV dataset file.

preprocess.py: Converts raw audio into normalized 3-channel Mel spectrogram images for ViT input.

model.py: Defines a Vision Transformer (ViT) model architecture fine-tuned for the classification task.

train.py: Handles dataset loading, augmentation, model training, validation, and saving.

llm.py: Interfaces with Groq’s OpenAI-compatible LLM API to get behavior explanations for predicted labels.

gradio_ui.py: Builds an interactive web interface for audio input, prediction, and explanation display.
