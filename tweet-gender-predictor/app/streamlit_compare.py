import streamlit as st
from transformers import DistilBertTokenizer, BertTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def load_model(model_name):
    model_path = f"C:/Desktop/Narratize Data/tweet-gender-predictor/models/{model_name}"
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return model

def predict_gender(tweet, model, tokenizer):
    inputs = tokenizer(tweet, return_tensors="pt", max_length=128, truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = F.softmax(logits, dim=1)
    prediction = torch.argmax(probabilities, dim=1).item()
    confidence = round(probabilities[0][prediction].item() * 100, 2)
    return 'Male' if prediction == 1 else 'Female', confidence

# Streamlit interface
st.title('Tweet Gender Prediction')

# Model selection
model_option = st.selectbox(
    'Choose a model:',
    ('distilbert-base-uncased', 'bert-base-uncased')
)

# Load tokenizer based on model selection
if model_option == 'distilbert-base-uncased':
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
else:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = load_model(model_option)

tweet = st.text_area("Enter a tweet:")
if st.button('Predict Gender'):
    prediction, confidence = predict_gender(tweet, model, tokenizer)
    st.write(f'Predicted Gender: {prediction} {confidence}%')