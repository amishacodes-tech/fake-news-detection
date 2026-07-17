# Fake News Detection using Machine Learning

## 📝 Project Overview
This project is a Machine Learning application designed to detect whether a news article is fake or real. It utilizes Natural Language Processing (NLP) techniques and TF-IDF Vectorization for text processing and classification.

## 🎯 Objectives
* Detect Fake News
* Detect Real News
* Learn NLP processing
* Apply Machine Learning algorithms

## ✨ Features
* User-friendly interface
* Real-time news classification
* Accurate predictions using pre-trained model

## 🛠 Installation
1. Clone Repository: `git clone <your-repo-link>`
2. Move to Project Folder: `cd <folder-name>`
3. Install Dependencies: `pip install -r requirements.txt`
4. Run Streamlit Application: `py -m streamlit run main.py`

## 📊 Output Files
* main.py
* model.pkl
* vectorizer.pkl
* Fake.csv
* True.csv

## 🚀 Future Improvements
* Multi-language News Detection
* URL-based Fake News Detection
* Advanced Sentiment Analysis
* Mobile Application

## 🛠 Technologies Used
* **Language:** Python
* **Web Framework:** Streamlit
* **Machine Learning:** Scikit-Learn
* **Data Processing:** Pandas, NumPy

## 💡 How it Works
1. **Data Loading:** The application loads the CSV files containing news data.
2. **Preprocessing:** Text data is cleaned, and TF-IDF vectorization is applied to convert text into numerical format.
3. **Prediction:** The trained Machine Learning model analyzes the input and predicts if the news is 'Fake' or 'Real'.

## 📂 Project Structure
```text
fakenewsdetection/
├── data/
│   ├── Fake.csv
│   └── True.csv
├── model/
├── main.py
├── train.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
└── README.md

## 📊 Model Performance
The performance of the model was evaluated using standard classification metrics:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **98%** |
| **Precision** | **98%** |
| **Recall** | **98%** |

*Note: The model demonstrates high reliability in distinguishing between real and fake news articles.*

## project output
![Project Output](Screenshot%202026-07-16%20201002.png)
## 📝 License
This project is for educational purposes only.

---

## 👤 Author
Developed by **Amisha**
