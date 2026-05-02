# Poetry NLP: Theme Classification in Shakespearean Sonnets

## Overview
This project explores whether simple NLP models can identify dominant themes in Shakespeare’s sonnets.

Given a poem as input, the model predicts its primary theme (beauty, time, or love).

---

## Dataset
- Shakespeare Sonnets (1–20)
- Each entry includes:
  - sonnet number
  - title
  - text
  - label

Labels:
- beauty
- time
- love

---

## Model
- TF-IDF vectorizer
- Logistic Regression classifier

Pipeline:
text → features → prediction

---

## Results
- Works on training data
- Low–moderate confidence (~0.35–0.60)
- Bias toward "time" for unclear cases

Example:

Sample poem:
Love looks not with the eyes, but with the mind

Predicted: time

---

## Limitations
- Small dataset (20 samples)
- Overlapping themes
- No proper evaluation split

---

## Future Work
- Add more sonnets
- Improve evaluation
- Explore interpretability

---

## Run

pip install pandas scikit-learn
python3 src/train.py

---

## Author
Afsah Buraaq
