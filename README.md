# Poetry NLP: Theme Classification in Shakespeare’s Sonnets

## Overview
This project explores whether simple NLP models can identify dominant themes in Shakespeare’s sonnets.

Given a sonnet as input, the model predicts its primary theme:
- beauty
- love
- time

The goal is not perfect accuracy, but to study how computational models behave on abstract, metaphor-rich literary text.

---

## Dataset

- Source: Shakespeare’s Sonnets (154 total)
- Parsed from raw text into structured format
- 40 sonnets manually labeled

### Labels
- beauty
- love
- time

Dataset pipeline:
- raw text → parsed corpus → labeled subset → merged dataset

---

## Model

- TF-IDF vectorization (unigrams + bigrams)
- Logistic Regression
- Class-balanced training
- Train/test split evaluation

---

## Results

### Classification Report

| Class  | Precision | Recall |
|--------|----------|--------|
| beauty | 0.50     | 0.67   |
| love   | 0.50     | 0.75   |
| time   | 0.50     | 0.20   |

Accuracy: ~0.50

---

## Key Insights

- Initial models over-predicted the **time** theme due to class imbalance
- Expanding and balancing the dataset improved recognition of **love**
- Shakespeare’s sonnets often blend themes, making strict classification difficult
- Model confidence remains low, reflecting inherent ambiguity in literary text

---

## Example Predictions

Sonnet 41 → love  
Sonnet 42 → love  
Sonnet 46 → beauty  
Sonnet 50 → time

---

## Project Structure

data/
  poems.csv
  poems_extended.csv
  shakespeare_full.csv

src/
  parse_sonnets.py
  merge_labels.py
  train.py

---

## Run the Model

pip install pandas scikit-learn  
python3 src/train.py

Optional CLI input:
python3 src/train.py "Love looks not with the eyes but with the mind"

---

## Limitations

- Small labeled dataset
- Overlapping literary themes
- No deep semantic understanding
- Performance sensitive to labeling choices

---

## Future Work

- Expand labeled dataset
- Improve evaluation methodology
- Explore interpretability of predictions
- Investigate multi-label classification for overlapping themes

---

## Author

Afsah Buraaq  
AI Undergraduate at MBZUAI  
Interested in NLP, language, and meaning in computational systems
