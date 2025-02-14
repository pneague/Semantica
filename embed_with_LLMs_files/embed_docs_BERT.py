import pandas as pd
import os
import torch
from transformers import BertTokenizer, BertModel

# Choose a BERT model. For example: "bert-base-uncased"
model_name = "bert-base-uncased"

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load the dataset
docs = pd.read_csv(os.path.join('dataset', 'doc.csv'), sep='\t')

# Open a log file for writing progress updates
log_file_path = 'progress_log.txt'

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
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    
    # Move inputs to GPU
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Pass through the encoder to get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use the mean of the token embeddings as the sentence embedding
    # outputs.last_hidden_state is of shape [batch_size, seq_length, hidden_size]
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding

# Apply the function to encode each row
docs["embedding"] = docs["Title"].apply(encode_text_with_progress)

# Save the embeddings to a pickle file
docs.to_pickle('embedded_docs_df.pkl')

