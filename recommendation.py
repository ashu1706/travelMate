import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load dataset
def load_data():
    csv_path = os.path.join(os.path.dirname(__file__), 'indian_travel_destinations.csv')
    df = pd.read_csv(csv_path)
    df["features"] = df["tags"] + " " + df["type"] + " " + df["region"] + " " + df["best_season"]
    return df

# Recommend destinations based on user input
def get_recommendations(user_input, top_n=5):
    df = load_data()

    vectorizer = TfidfVectorizer()
    feature_matrix = vectorizer.fit_transform(df["features"])

    input_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(input_vec, feature_matrix)

    indices = similarity.argsort()[0][-top_n:][::-1]
    recommended = df.iloc[indices][["name", "region", "type", "tags"]]

    return recommended.to_dict(orient="records")

if __name__ == "__main__":
    user_input = "beach summer low-budget"
    recommendations = get_recommendations(user_input)
    print("Recommended Destinations:")
    for rec in recommendations:
        print(f"Name: {rec['name']}, Region: {rec['region']}, Type: {rec['type']}, Tags: {rec['tags']}")
