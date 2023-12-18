import json
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from transformers import AdamW, get_linear_schedule_with_warmup

# Function to load data from .jl file
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    return pd.DataFrame(data)

# Load and preprocess data
data_file = r'C:\Users\Peant\OneDrive\Desktop\Narratize Data\tweet-gender-predictor\data\data.jl'
df = load_data(data_file)
df['gender_label'] = df['gender'].map({'F': 0, 'M': 1})
train_df, val_df = train_test_split(df, test_size=0.2)

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize function
def tokenize_function(texts):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=128)

train_encodings = tokenize_function(train_df["text"].tolist())
val_encodings = tokenize_function(val_df["text"].tolist())

# Dataset class
class TweetDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TweetDataset(train_encodings, train_df["gender_label"].tolist())
val_dataset = TweetDataset(val_encodings, val_df["gender_label"].tolist())

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Check for CUDA and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model and move to device
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.to(device)

# Parameters
num_epochs = 3
learning_rate = 5e-5
eps = 1e-8

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, eps=eps)

# Total number of training steps
total_steps = len(train_loader) * num_epochs

# Scheduler
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0, 
                                            num_training_steps=total_steps)


# Training loop
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0

    for i, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        
        loss = outputs.loss
        total_train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Print progress every 10 batches
        if (i + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Batch {i + 1}/{len(train_loader)}, Training Loss: {loss.item():.4f}")

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch {epoch + 1} | Average Training Loss: {avg_train_loss}")

    # Validation loop
    model.eval()
    total_eval_loss = 0

    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        total_eval_loss += loss.item()

    avg_val_loss = total_eval_loss / len(val_loader)
    print(f"Epoch {epoch + 1} | Average Validation Loss: {avg_val_loss}")


# Save the fine-tuned model
model.save_pretrained(r'C:\Users\Peant\OneDrive\Desktop\Narratize Data\tweet-gender-predictor\models\bert-base-uncased')