# ADS509 Final Project  
## Emotion Classification Using Transformer Models

## 📌 Project Overview

This project develops and evaluates machine learning models for **multi-class emotion classification** using Twitter text data. The goal is to classify tweets into emotion categories (e.g., joy, love, anger, fear, sadness, surprise) using both baseline and transformer-based approaches.

The final notebook included in this repository is:

`ADS509FinalProject.ipynb`

The dataset used is:

`tweet_emotions.csv`

---

## 📂 Repository Structure

```

├── ADS509FinalProject.ipynb      # Final cleaned notebook
├── tweet_emotions.csv            # Emotion-labeled Twitter dataset
└── README.md                     # Project documentation

```

---

## 📊 Dataset

**File:** `tweet_emotions.csv`  
**Shape:** 40,000 rows × 3 columns  

**Columns:**

- `tweet_id` – Unique tweet identifier  
- `sentiment` – Emotion label (target variable)  
- `content` – Tweet text  

The dataset contains labeled tweets across multiple emotion categories.

---

## 🧠 Methods

This project compares two modeling approaches:

### 1️⃣ Baseline Model  
- Traditional machine learning pipeline  
- Text preprocessing (cleaning, tokenization)  
- TF-IDF vectorization  
- Logistic Regression classifier  

### 2️⃣ Transformer Model  
- Fine-tuned DistilBERT  
- Pretrained transformer architecture  
- HuggingFace Transformers library  
- End-to-end fine-tuning for multi-class classification  

---

## ⚙️ Requirements

Recommended environment:

- Python 3.9+
- Google Colab or Jupyter Notebook

Main libraries used:

```

pandas
numpy
scikit-learn
matplotlib
seaborn
torch
transformers
datasets

```

To install core dependencies:

```

pip install pandas numpy scikit-learn matplotlib seaborn torch transformers datasets

```

---

## 🚀 How to Run

1. Clone the repository:
```

git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

````

2. Ensure `tweet_emotions.csv` is located in the root directory.

3. Open the notebook:
- In Jupyter:
  ```
  jupyter notebook ADS509FinalProject.ipynb
  ```
- Or upload to Google Colab.

4. Run all cells sequentially from top to bottom.

---

## 📈 Results Summary

- The baseline TF-IDF + Logistic Regression model provides a strong performance benchmark.
- The fine-tuned DistilBERT model improves classification performance by better capturing contextual meaning.
- Transformer models handle semantic nuance and short-form social media text more effectively.

---

## 🎯 Key Takeaways

- Transformer-based models outperform traditional NLP pipelines for emotion classification.
- Fine-tuning pretrained models reduces the need for manual feature engineering.
- Emotion overlap (e.g., joy vs. love) and class imbalance present modeling challenges.

---

## 📝 Authors

Akshat Patni, Kirsten Drennen, Alli McKernan  
ADS509 – Final Project  
Emotion Classification with Transformers
````
