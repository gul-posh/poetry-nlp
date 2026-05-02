import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load dataset
df_all = pd.read_csv("data/poems_extended.csv")
df = df_all.dropna(subset=["label"]).copy()

X = df["text"]
y = df["label"]

print("Total labeled samples:", len(df), "\n")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("Train size:", len(X_train))
print("Test size:", len(X_test), "\n")

# Vectorize
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)

print("Evaluation:\n")
print(classification_report(y_test, y_pred, zero_division=0))

# Predict on unseen
df_unseen = df_all[df_all["label"].isna()].head(10)

print("\nPredictions on unseen:\n")

for _, row in df_unseen.iterrows():
    vec = vectorizer.transform([row["text"]])
    pred = model.predict(vec)[0]
    print(f"Sonnet {row['sonnet_number']} → {pred}")
