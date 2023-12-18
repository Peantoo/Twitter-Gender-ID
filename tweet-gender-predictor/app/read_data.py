import json
import random

def sample_tweets(file_path, sample_size=100):
    with open(file_path, 'r') as file:
        tweets = [json.loads(line) for line in file]

    # Ensure sample size is not larger than the number of available tweets
    sample_size = min(sample_size, len(tweets))

    sampled_tweets = random.sample(tweets, sample_size)
    
    for tweet in sampled_tweets:
        print(json.dumps(tweet, indent=2))

# Usage
data_file = r"C:\Users\Peant\OneDrive\Desktop\Narratize Data\tweet-gender-predictor\data\data.jl"
sample_tweets(data_file)