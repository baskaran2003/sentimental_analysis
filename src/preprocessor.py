from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib
import os


def preprocess_data(df, save_path="models/"):
    """Preprocess text data and encode labels."""
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['review'])  # Assuming 'review' column

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['sentiment'])  # Convert positive/negative/neutral to numbers

    # Save TF-IDF vectorizer and Label Encoder
    os.makedirs(save_path, exist_ok=True)
    joblib.dump(tfidf, os.path.join(save_path, "tfidf_vectorizer.pkl"))
    joblib.dump(label_encoder, os.path.join(save_path, "label_encoder.pkl"))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
