# Gradio UI: Cat Meow Translator
# =========================================

import gradio as gr
import torchaudio
import torch
import timm
from torchvision import transforms
from PIL import Image
import io

# Load the pretrained ViT model weights and set to evaluation mode
model.load_state_dict(torch.load("best_vit.pth"))
model.eval()

# Preprocess function: converts raw audio into a 3-channel Mel spectrogram image
def preprocess(audio_path):
    # Load the audio waveform and sample rate
    waveform, sr = torchaudio.load(audio_path)
    
    # Convert to mono by averaging channels if stereo
    waveform = waveform.mean(dim=0, keepdim=True)
    
    # Define fixed length (2 seconds at 16kHz)
    num_samples = 16000 * 2
    
    # Pad or trim the waveform to the fixed length
    waveform = torch.nn.functional.pad(waveform, (0, max(0, num_samples - waveform.size(1))))[:, :num_samples]
    
    # Compute Mel spectrogram (frequency representation)
    mel = torchaudio.transforms.MelSpectrogram(sr, n_fft=1024, n_mels=128)(waveform)
    
    # Convert amplitude to decibels (log scale)
    mel = torchaudio.transforms.AmplitudeToDB()(mel)
    
    # Compute first and second order deltas (derivatives) of the Mel spectrogram
    delta = torchaudio.functional.compute_deltas(mel)
    delta2 = torchaudio.functional.compute_deltas(delta)
    
    # Normalize each channel to [0, 1]
    norm = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
    
    # Stack normalized mel, delta, and delta2 channels to form a 3-channel image
    stacked = torch.cat([norm(mel), norm(delta), norm(delta2)], dim=0)
    
    # Convert tensor to PIL image for feeding into ViT model
    img = transforms.ToPILImage()(stacked)
    return img

# Main function: takes uploaded audio, predicts label, and explains via LLM
def predict_and_explain(audio):
    # Save the uploaded audio file temporarily
    audio_path = "temp_input.wav"
    with open(audio_path, "wb") as f:
        f.write(audio.read())

    # Preprocess audio into spectrogram image
    img = preprocess(audio_path)
    
    # Define transformation pipeline to resize and convert image to tensor
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    
    # Apply transforms and add batch dimension, then move tensor to the appropriate device (CPU/GPU)
    input_tensor = tf(img).unsqueeze(0).to(device)

    # Inference: disable gradient calculation for efficiency
    with torch.no_grad():
        out = model(input_tensor)                  # Forward pass
        pred_idx = out.argmax(1).item()            # Get predicted class index
        # Map index back to label string
        pred_label = [k for k,v in label2id.items() if v == pred_idx][0]

    # Compose prompt for LLM to explain the predicted label
    prompt = f"""
    You are an expert cat behaviorist.
    The sound label is: '{pred_label}'.
    Explain what this sound probably means and what the owner should do.
    """

    # Query the LLM (Groq-hosted) for explanation
    response = client.chat.completions.create(
        model="llama3-70b-8192",                  # Model name
        messages=[
            {"role": "system", "content": "You are a helpful cat behavior assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    # Extract explanation text from response
    explanation = response.choices[0].message.content

    # Return formatted markdown with predicted label and explanation
    return f"Predicted Label: **{pred_label}**\n\n{explanation}"

# Define and launch the Gradio interface
gr.Interface(
    fn=predict_and_explain,                        # Function to run on audio input
    inputs=gr.Audio(type="filepath"),              # Input is an audio file upload (filepath)
    outputs="markdown",                             # Output rendered as markdown text
    title="Chat With Your Cat",                  # Interface title
    description="Upload your cat's meow. The model classifies it and the LLM explains what your cat wants!"  # Description
).launch()
