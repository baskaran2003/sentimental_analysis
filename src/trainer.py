
from sklearn.ensemble import RandomForestClassifier
import mlflow 
import mlflow.sklearn
import os

def train_model(X_train, y_train):
    """Train Random Forest model and log with MLflow."""
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        
        # Log model
        mlflow.sklearn.log_model(model, "sentiment_model")
        mlflow.log_param("n_estimators", 100)
        
    return model