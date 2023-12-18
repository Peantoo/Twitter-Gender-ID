# time_extraction.py
import pandas as pd
import dateutil.parser
from pymongo import MongoClient

def get_tweet_times():
    client = MongoClient('localhost', 27017)
    db = client['Gender-Prediction']
    tweets = db['Tweets']
    
    pipeline = [
        {"$lookup": {
            "from": "Truth Data",
            "localField": "user_id",
            "foreignField": "user_id",
            "as": "user_info"
        }},
        {"$unwind": "$user_info"},
        {"$project": {
            "created_at": 1,
            "gender": "$user_info.gender"
        }}
    ]
    result = tweets.aggregate(pipeline)
    df = pd.DataFrame(list(result))
    
    # Parse the 'created_at' column to datetime and extract hour
    df['hour'] = df['created_at'].apply(lambda x: dateutil.parser.parse(x).hour)

    return df

def create_processed_file():
    df = get_tweet_times()
    output_path = "C:/Desktop/Narratize Data/tweet-gender-predictor/data/processed_tweets.json"
    df.to_json(output_path, orient='records', lines=True)

if __name__ == "__main__":
    create_processed_file()
