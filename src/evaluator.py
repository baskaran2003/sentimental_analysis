from sklearn.metrics import accuracy_score
import mlflow 


def evaluate_model(model, X_test, y_test):
    """Evaluate model and log accuracy to MLflow."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    mlflow.log_metric("accuracy", acc)
    