import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


# Function to load data from .jl file
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    return pd.DataFrame(data)

# Load and preprocess data
data_file = r'C:\Desktop\Narratize Data\tweet-gender-predictor\data\data.jl'
df = load_data(data_file)
df['gender_label'] = df['gender'].map({'F': 0, 'M': 1})
train_df, val_df = train_test_split(df, test_size=0.2)

tk_model = 'distilbert-base-uncased'
# Initialize tokenizer for DistilBERT
tokenizer = DistilBertTokenizer.from_pretrained(tk_model)

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

# Check for CUDA and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Initialize model and move to device
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# New training arguments
batch_size = 16
logging_steps = len(train_dataset) // batch_size
model_name = "distilbert-finetuned-gender"
training_args = TrainingArguments(
    output_dir=model_name,
    num_train_epochs=4,
    learning_rate=3e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    disable_tqdm=False,
    logging_steps=logging_steps,
    log_level="error"
)

# New compute_metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

# Using Trainer for training
trainer = Trainer(
    model=model, 
    args=training_args, 
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)
trainer.train()


# Save the fine-tuned model
model.save_pretrained(r'C:\Desktop\Narratize Data\tweet-gender-predictor\models\distilbert-base-uncased')