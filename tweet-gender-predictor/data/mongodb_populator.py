import os
import json
from pymongo import MongoClient

# MongoDB setup
client = MongoClient('localhost', 27017)  # Connect to MongoDB
db = client['Gender-Prediction']  # Use (or create) a database named 'twitter_data'
tweets_collection = db['Tweets']  # Use (or create) a collection named 'tweets'

# Directory containing your JSON files
jl_directory = r'C:\Desktop\Narratize Data\tweet-gender-predictor\data\tweet_files'  # Update with the path to your JSON files

# Iterate through each file in the directory and insert into MongoDB
for filename in os.listdir(jl_directory):
    if filename.endswith('.jl'):  # Check if the file is a JSON Lines file
        file_path = os.path.join(jl_directory, filename)
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    tweet = json.loads(line)  # Parse JSON object from each line
                    tweets_collection.insert_one(tweet)  # Insert the tweet into the collection
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON in file {filename}, line: {line}\nError: {e}")

print("Data import complete.")
