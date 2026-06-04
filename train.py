import mlflow
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
mlflow.autolog()
# 1. Siapkan data sederhana
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Jalankan MLflow Tracking
with mlflow.start_run():
    # Membangun model ML
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # Hitung metrik
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # Log ke MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "random_forest_model")
    
    print(f"Eksperimen selesai! Akurasi model: {accuracy}")