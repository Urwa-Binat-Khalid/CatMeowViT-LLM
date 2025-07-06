# imports
# IPython helpers for audio playback and markdown rendering
from IPython.display import Audio, display, Markdown

# Pandas for data loading, random for sampling
import pandas as pd
import random

# OpenAI Python client for Groq LLM calls
from openai import OpenAI  


# =========================================
# Connect to Groq LLM API
# =========================================

# Initialize OpenAI client with Groq's base URL
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",  # Groq-compatible endpoint
    api_key=""  # Add your Groq API key here!
)


# =========================================
# Load the balanced dataset
# =========================================

CSV_PATH = "/kaggle/working/balanced_dataset.csv"
df = pd.read_csv(CSV_PATH)


# =========================================
# Pick a random audio file & label
# =========================================

random_row = df.sample(1).iloc[0]  # Random row
random_path = random_row['file_path']  # Audio path
random_label = random_row['label']     # Label for the sound

# Show which file was picked
print(f" Random sample: '{random_label}'\nPath: {random_path}")

# Play the audio inside notebook
display(Audio(filename=random_path))


# =========================================
#  Query the LLM for an explanation
# =========================================

print("\n LLM Response:\n")

# Prompt for the LLM: explain what the sound means & advice for the owner
prompt = f"""
You are an expert cat behaviorist.
The sound label is: '{random_label}'.
Explain in detail what this sound probably means and what the owner should do.
"""

# Make the LLM chat request
response = client.chat.completions.create(
    model="llama3-70b-8192",  # Groq-supported model name
    messages=[
        {"role": "system", "content": "You are a helpful assistant for cat owners."},
        {"role": "user", "content": prompt}
    ]
)

# Extract text from the response
llm_answer = response.choices[0].message.content

# Show the LLM's answer nicely
display(Markdown(f"**LLM says:** {llm_answer}"))
