import json
import re
import os
import glob

def read_manifest(file_path):
    manifest_data = {}
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            user_id = str(data.get('user_id_str'))  # Convert user_id to string
            if user_id:
                manifest_data[user_id] = {
                    'num_tweets': data.get('num_tweets', 0),
                    'gender': data.get('gender_human', 'unknown')
                }
    return manifest_data

def is_english(text):
    # Regex pattern to match non-English characters and symbols
    # It allows English letters (both cases), numbers, and basic punctuation
    non_english_pattern = re.compile('[^a-zA-Z0-9 .,!?]')
    return not non_english_pattern.search(text)

def clean_tweet(text):
    # Remove emojis and other non-text content
    text = re.sub(r'[^\w\s]', '', text)
    # Normalize text (e.g., lowercase, remove extra whitespace)
    text = ' '.join(text.lower().strip().split())
    return text

def is_informative(tweet_text):
    # Check if the tweet is too short - "yea" probably isn't useful.
    if len(tweet_text.split()) <= 2:
        return False

    # Check if the tweet is just a mention or URL
    if re.match(r'^(@\w+ ?)+$', tweet_text) or 'http' in tweet_text:
        return False

    return True

def process_tweets(file_name, manifest_data):
    processed_tweets = []

    with open(file_name, 'r') as file:
        for line in file:
            tweet = json.loads(line)
            user_id = str(tweet['document']['user']['id'])
            tweet_text = tweet['document']['text']

            if tweet_text and is_english(tweet_text) and is_informative(tweet_text):
                cleaned_text = clean_tweet(tweet_text)
                if cleaned_text:
                    tweet_data = {'user_id': user_id, 'text': cleaned_text}
                    # Append metadata from manifest
                    tweet_data.update(manifest_data.get(user_id, {}))
                    processed_tweets.append(tweet_data)

    return processed_tweets

manifest_file = r'C:\Users\Peant\OneDrive\Desktop\Narratize Data\tweet-gender-predictor\data\manifest.jl'
manifest_data = read_manifest(manifest_file)

tweet_files_directory = r'C:\Users\Peant\OneDrive\Desktop\Narratize Data\tweet-gender-predictor\data\tweet_files'
combined_tweets = []
for file_name in glob.glob(os.path.join(tweet_files_directory, '*.jl')):
    combined_tweets.extend(process_tweets(file_name, manifest_data))


def process_all_tweets(directory, manifest_data):
    all_tweets = []
    for file_name in glob.glob(os.path.join(directory, '*.jl')):
        all_tweets.extend(process_tweets(file_name, manifest_data))
    return all_tweets

# Process all tweets and combine them
combined_tweets = process_all_tweets(tweet_files_directory, manifest_data)

# Write combined tweets to a new file
output_file = r"C:\Users\Peant\OneDrive\Desktop\Narratize Data\tweet-gender-predictor\data\data.jl"
with open(output_file, 'w') as outfile:
    for tweet in combined_tweets:
        json.dump(tweet, outfile)
        outfile.write('\n')

# Example usage
print(f"Processed {len(combined_tweets)} tweets.")