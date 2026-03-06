# 📰 Fake News Detection System

A **Machine Learning pipeline** that classifies news articles as **Real** or **Fake** using **TF-IDF vectorization** and two classifiers:

* **Logistic Regression**
* **Multinomial Naive Bayes**

The system can be trained on **full article text** or **news headlines (titles)** and supports **unigrams, bigrams, and trigrams** to capture phrase patterns more effectively.

You can interact with the trained model through:

* 🖥 **Command Line Interface** – `fake_news_detector.py`
* 🌐 **Streamlit Web App** – `app.py`
* 🪟 **Tkinter Desktop GUI (optional)** – `gui.py`

---

# 📁 Dataset

This project uses the **Fake and Real News Dataset** from Kaggle:

https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset

Download the dataset and place the files inside a folder named:

```
dataset/
```

Required files:

```
dataset/
│
├── Fake.csv     # Fake news articles
└── True.csv     # Real news articles
```

Both files must contain:

* `text` → Full article
* `title` → News headline

During preprocessing, the script automatically creates a **label column**:

```
0 → Fake News
1 → Real News
```

---

# ⚙️ Installation

### 1️⃣ Clone the Repository

```
git clone https://github.com/Tarun524-code/fake-news-detection
cd fake-news-detection
```

### 2️⃣ Install Required Packages

It is recommended to use a **virtual environment**.

```
pip install -r requirements.txt
```

### 3️⃣ Add Dataset

Place `Fake.csv` and `True.csv` inside the `dataset/` folder.

---

# 🚀 Usage

## 1️⃣ Train the Models (Command Line)

Run the training pipeline:

```
python fake_news_detector.py --train
```

Default training settings:

* Uses **full article text**
* Uses **unigrams + bigrams (1,2)**

---

### Train Using Headlines Instead

```
python fake_news_detector.py --train --feature-column title --ngram-range 1 3
```

This trains using:

* **News headlines**
* **Unigrams + Bigrams + Trigrams**

---

### Training Process

The script performs:

* **TF-IDF vectorization**
* **GridSearchCV hyperparameter tuning**
* **80/20 train-test split**
* **Model evaluation**
* **Model saving**

Saved files:

```
models/
│
├── best_model.pkl
├── logistic_regression_model.pkl
├── naive_bayes_model.pkl
├── tfidf_vectorizer.pkl
└── training_metadata.json
```

---

# 📊 Evaluate the Trained Models

After training:

```
python fake_news_detector.py --evaluate
```

Evaluation includes:

* Accuracy
* Precision / Recall / F1-score
* ROC-AUC score
* Confusion Matrix

The script automatically loads **training metadata** to ensure evaluation uses the same preprocessing settings.

---

# 🔍 Predict a News Article (Command Line)

```
python fake_news_detector.py --predict "Your news headline or article text here"
```

Example output:

```
Prediction: REAL NEWS
Confidence: 0.97
```

---

# 🌐 Streamlit Web App (Recommended)

A modern **browser interface** for the model.

Run:

```
streamlit run app.py
```

If Streamlit is not in PATH:

```
python -m streamlit run app.py
```

### Features

* Large text area for input
* Real-time prediction
* Confidence score
* Color result box

  * 🟢 Green → Real
  * 🔴 Red → Fake
* Probability bar chart
* Word importance analysis (Logistic Regression)

---

# 🧠 Command Line Arguments

| Argument                        | Description                   |
| ------------------------------- | ----------------------------- |
| `--train`                       | Train models from scratch     |
| `--evaluate`                    | Evaluate trained models       |
| `--predict TEXT`                | Predict a single news article |
| `--feature-column {text,title}` | Choose input column           |
| `--ngram-range N M`             | Set n-gram range              |

Example:

```
python fake_news_detector.py --train --feature-column title --ngram-range 1 3
```

---

# 🗂️ Project Structure

```
fake-news-detection/
│
├── dataset/
│   ├── Fake.csv
│   └── True.csv
│
├── models/
│
├── fake_news_detector.py
├── app.py
├── gui.py
├── requirements.txt
└── README.md
```

---

# 🧪 Metadata Handling

To avoid mismatches between **training and evaluation**, the project stores metadata:

```
models/training_metadata.json
```

Example:

```
{
  "feature_column": "title",
  "ngram_range": [1, 3]
}
```

During evaluation, this metadata is automatically loaded so preprocessing remains **consistent**.

---

# 📈 Performance

Typical performance on the dataset:

| Input Type     | Accuracy |
| -------------- | -------- |
| Full Articles  | ~99%     |
| Headlines Only | ~95-96%  |

Both classifiers perform well, but **Logistic Regression usually performs slightly better**.

---

# 🛠️ Customisation

You can easily extend the project:

### Add More Models

Edit the training section in:

```
fake_news_detector.py
```

### Change Test Split

Modify:

```
test_size parameter in split_data()
```

### Adjust TF-IDF Settings

Edit the function:

```
preprocess_text()
```

---

# 📌 Requirements

Main libraries used:

```
pandas
numpy
scikit-learn
matplotlib
seaborn
streamlit
joblib
```

Install them with:

```
pip install -r requirements.txt
```

---

# 📜 License

This project is for **educational and research purposes**.

---

# ⭐ If You Like This Project

Consider giving it a **star ⭐ on GitHub**.
