# preprocess.py

import os
import torch
import torchaudio
from torchvision import transforms
from tqdm import tqdm
import pandas as pd

from config import CSV_PATH, OUT_DIR, SAMPLE_RATE, DURATION

class WaveformAugment:
    """
    Simple waveform augmentation class.
    Adds random noise to the waveform for slight data augmentation.
    """
    def __init__(self, sr):
        self.sr = sr

    def __call__(self, waveform):
        # Add Gaussian noise to the waveform
        return waveform + torch.randn_like(waveform) * 0.002

def save_specs():
    """
    Convert audio files to mel spectrogram images.
    For each audio file:
      - Resample to target SAMPLE_RATE if needed.
      - Convert to mono.
      - Apply augmentation (noise).
      - Pad or crop to fixed DURATION.
      - Compute mel spectrogram and its deltas.
      - Normalize and stack them.
      - Save as RGB image in OUT_DIR.
    """
    # Load the balanced CSV
    df = pd.read_csv(CSV_PATH)
    sr, duration = SAMPLE_RATE, DURATION
    num_samples = int(sr * duration)

    # Create waveform augmenter
    waug = WaveformAugment(sr)

    # Make sure output directory exists
    os.makedirs(OUT_DIR, exist_ok=True)

    # Iterate over each audio file in the DataFrame
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        f, label = row['file_path'], row['label']

        # Create output subfolder for the label
        out_dir = os.path.join(OUT_DIR, label)
        os.makedirs(out_dir, exist_ok=True)

        # Load audio file
        waveform, fs = torchaudio.load(f)

        # Resample if needed
        if fs != sr:
            waveform = torchaudio.transforms.Resample(fs, sr)(waveform)

        # Convert to mono by averaging channels
        waveform = waveform.mean(dim=0, keepdim=True)

        # Apply waveform augmentation
        waveform = waug(waveform)

        # Pad or truncate to fixed number of samples
        waveform = torch.nn.functional.pad(
            waveform, (0, max(0, num_samples - waveform.size(1)))
        )[:, :num_samples]

        # Compute mel spectrogram
        mel = torchaudio.transforms.MelSpectrogram(
            sr, n_fft=1024, n_mels=128
        )(waveform)

        # Convert to decibels
        mel = torchaudio.transforms.AmplitudeToDB()(mel)

        # Compute deltas (1st and 2nd derivatives)
        delta = torchaudio.functional.compute_deltas(mel)
        delta2 = torchaudio.functional.compute_deltas(delta)

        # Normalize each channel to [0, 1]
        norm = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)

        # Stack original, delta, and delta2 into 3-channel image
        stacked = torch.cat([norm(mel), norm(delta), norm(delta2)], dim=0)

        # Convert tensor to PIL Image and save as PNG
        img = transforms.ToPILImage()(stacked)
        img.save(os.path.join(out_dir, f"{idx}.png"))
