import streamlit as st
from transformers import DistilBertTokenizer, BertTokenizer, DistilBertForSequenceClassification, DistilBertModel, DistilBertConfig
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
#@st.cache(allow_output_mutation=True)
def load_model(model_name):
    model_path = f"C:/Desktop/Narratize Data/tweet-gender-predictor/models/{model_name}"
    # Load the base DistilBert Model
    config = DistilBertConfig.from_pretrained(model_path, output_attentions=True)
    model = DistilBertModel.from_pretrained(model_path, config=config)
    # Add a classification layer
    classifier = torch.nn.Linear(config.dim, 2)  # Assuming binary classification
    return model, classifier

def predict_gender(tweet, model, classifier, tokenizer):
    inputs = tokenizer(tweet, return_tensors="pt", max_length=128, truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = classifier(outputs.last_hidden_state[:, 0, :])
        attentions = outputs.attentions

    probabilities = F.softmax(logits, dim=1)
    prediction = torch.argmax(probabilities, dim=1).item()
    confidence = round(probabilities[0][prediction].item() * 100, 2)
    return 'Male' if prediction == 1 else 'Female', confidence, attentions

def plot_attention(attentions, tokens):
    # Create a new figure and axis for the plot
    fig, ax = plt.subplots()
    
    # Plotting the first head of the last layer attention
    attention = attentions[-1][0, 0].detach().numpy()
    sns.heatmap(attention, xticklabels=tokens, yticklabels=tokens, ax=ax)
    
    # Return the figure to be used in st.pyplot()
    return fig

# Streamlit interface
st.title('Tweet Gender Prediction')

# Model selection
model_option = 'distilbert-base-uncased'  # Currently, only DistilBERT is set up for this

# Load tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained(model_option)
model, classifier = load_model(model_option)

tweet = st.text_area("Enter a tweet:")
if st.button('Predict Gender'):
    tokens = tokenizer.tokenize(tweet)
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    prediction, confidence, attentions = predict_gender(tweet, model, classifier, tokenizer)
    st.write(f'Predicted Gender: {prediction} {confidence}%')

    if attentions:
        fig = plot_attention(attentions, tokens)
        st.pyplot(fig)
