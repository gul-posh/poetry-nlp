import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("data/poems.csv")

print("Dataset loaded.\n")

# Features and labels
X = df["text"]
y = df["label"]

print("Label counts:")
print(y.value_counts())
print()

# Build model
model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000))
])

# Train model
model.fit(X, y)

print("Model trained.\n")

# Predict on training data
predictions = model.predict(X)
probabilities = model.predict_proba(X)
classes = model.classes_

print("Training predictions:\n")
for title, true_label, pred_label, probs in zip(df["title"], y, predictions, probabilities):
    top_index = probs.argmax()
    confidence = probs[top_index]
    print(title)
    print("  True:", true_label)
    print("  Pred:", pred_label)
    print("  Confidence:", round(confidence, 3))
    print()

# Test on a new poem
sample_poem = """
Love looks not with the eyes, but with the mind,
And therefore is winged Cupid painted blind.
"""

sample_probs = model.predict_proba([sample_poem])[0]
sample_pred = model.predict([sample_poem])[0]

print("Sample poem prediction:")
print("  Predicted label:", sample_pred)
print("  Class probabilities:")
for label, prob in zip(classes, sample_probs):
    print(f"    {label}: {prob:.3f}")
