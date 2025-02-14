import pandas as pd
import os
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

# Choose the smallest LLaMA-based model available.
# This is an example using Llama 2 7B. Adjust as needed.
model_name = "meta-llama/Llama-3.2-1B"

# Load the LLaMA tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # adjust if needed
    device_map="auto"           # automatically place on GPU if available
)

# Move the model to GPU if not already done by device_map
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the dataset
docs = pd.read_csv(os.path.join('dataset', 'doc.csv'), sep='\t')

# Open a log file for writing progress updates
log_file_path = 'progress_log_llama.txt'

# Counter for progress tracking
progress = {"counter": 0}

# Ensure the log file is empty at the start
with open(log_file_path, 'w') as log_file:
    log_file.write("Progress Log Initialized\n")

# Function to log progress to a file
def log_progress(message):
    with open(log_file_path, 'a') as log_file:
        log_file.write(message + "\n")

# Function to encode text with progress monitoring
def encode_text_with_progress(text):
    # Increment the counter
    progress["counter"] += 1
    
    # Log progress every 100 rows
    if progress["counter"] % 100 == 0:
        log_progress(f"{progress['counter']} texts encoded.")
    
    # Tokenize the input text
    # LLaMA tokenizer uses "<s>" as bos token; ensure truncation to a reasonable length
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    )
    
    # Move inputs to GPU
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Pass through the model to get hidden states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # outputs.hidden_states is a tuple of all hidden states
        # outputs.last_hidden_state is the last layer's hidden state
        
    # Extract the last hidden state
    # Shape: [batch_size, seq_length, hidden_size]
    last_hidden_state = outputs.hidden_states[-1]
    
    # Use the mean of the token embeddings as the sentence embedding
    embedding = last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding

# Apply the function to encode each row
docs["embedding"] = docs["Title"].apply(encode_text_with_progress)

# Save the embeddings to a pickle file
docs.to_pickle('embedded_docs_df_llama2.pkl')

