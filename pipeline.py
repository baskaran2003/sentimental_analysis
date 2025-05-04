import os
import mlflow

from src.data_loader import load_data
from src.preprocessor import preprocess_data
from src.trainer import train_model
from src.evaluator import evaluate_model

def run_pipeline(data_path="data/Reviews.csv", model_dir="models/"):
    
    # 1. Load Data
    print("📥 Loading data...")
    df = load_data(data_path)

    if 'review' not in df.columns or 'sentiment' not in df.columns:
        raise ValueError("Dataset must contain 'review' and 'sentiment' columns.")

    # 2. Preprocess Data
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(df, save_path=model_dir)

    # 3. Train Model
    print("Training model...")
    model = train_model(X_train, y_train)

    # 4. Evaluate Model
    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)

    print("Pipeline complete.")

if __name__ == "__main__":
    
    mlflow.set_experiment("SentimentAnalysisPipeline")

    run_pipeline()
