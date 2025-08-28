from data import load_data
from preprocess import preprocess_data
from train import train_model
from fairness import evaluate_fairness

print("Loading data...")
df = load_data()

print("Preprocessing data...")
X_train, X_test, y_train, y_test = preprocess_data(df)

print("Training model...")
model, acc = train_model(X_train, y_train, X_test, y_test)
print(f"Model Accuracy: {acc:.2f}")

print("Evaluating fairness...")
fairness_score = evaluate_fairness(df)
print(f"Disparate Impact: {fairness_score:.2f}")