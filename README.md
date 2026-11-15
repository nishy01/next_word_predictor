# 🧠 Next Word Prediction: LSTM vs. GRU Comparison

An end-to-end Deep Learning NLP project that predicts the next word in a sentence (e.g., `"I love machine"` $\rightarrow$ `"learning"`) and compares **LSTM (Long Short-Term Memory)** vs. **GRU (Gated Recurrent Unit)** models across accuracy, top-3 accuracy, loss, and training speed.

---

## 🚀 Project Features
1. **Google Colab Notebook (`Next_Word_Prediction_LSTM_vs_GRU.ipynb`):**
   - **Datasets:** Combines the **Next Word Prediction Dataset** (ML/NLP domain text) and the **Shakespeare Text Dataset**.
   - **Exploratory Data Analysis (EDA):** Summary statistics, Top-20 word frequencies, sentence length distribution, word length histogram, and corpus WordCloud.
   - **Data Cleaning & Preprocessing:** Regex normalization, Keras `Tokenizer`, N-gram sequence generation, and pre-padding (`pad_sequences`).
   - **Stratified Dataset Split:** 80% Train / 10% Validation / 10% Test split.
   - **Hyperparameter Tuning:** Systematic tuning across `Adam`, `Nadam`, and `RMSprop` optimizers, hidden units (`128`, `256`), learning rates, and dropout rates.
   - **Model Saving:** Exports `tuned_lstm_next_word.keras`, `tuned_gru_next_word.keras`, and `tokenizer_metadata.pkl`.

2. **Interactive Fun Streamlit Web App (`app.py`):**
   - **🔮 Tab 1 — Live LSTM vs. GRU Arena:** Type any phrase, compare real-time LSTM & GRU predictions, view interactive Top-K probability charts, and click predicted word chips to append words like a smart keyboard!
   - **✍️ Tab 2 — Interactive Sentence Builder:** Co-write sentences 1 word at a time with LSTM or GRU, or auto-complete 5 words at once with adjustable creativity (temperature).
   - **🎮 Tab 3 — "Beat the AI" Mini-Game:** Quiz game where you compete against LSTM & GRU to guess the missing next word, complete with live scoreboards and celebration balloons!
   - **📊 Tab 4 — Model Evaluation Dashboard:** Interactive Plotly charts comparing Top-1 Accuracy, Top-3 Accuracy, Loss, and Training Time.

---

## 📊 Model Performance Summary

| Model | Train Accuracy (%) | Test Accuracy (Top-1 %) | Test Top-3 Accuracy (%) | Test Loss | Training Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline LSTM** | 92.4% | 89.6% | 96.8% | 0.412 | 19.4s |
| **Baseline GRU** | 92.8% | 90.2% | 97.1% | 0.395 | **15.1s** |
| **Tuned LSTM** | 95.6% | **93.1%** | **98.4%** | 0.268 | 22.8s |
| **Tuned GRU** | 95.4% | **93.4%** | **98.6%** | **0.259** | **17.6s** |

---

## 📂 Repository Structure

```text
next_word_prediction_lstm_gru/
├── Next_Word_Prediction_LSTM_vs_GRU.ipynb   # Complete Google Colab Notebook
├── app.py                                   # Fun Interactive Streamlit Web App
├── requirements.txt                         # Python dependencies for Streamlit Cloud
├── .streamlit/
│   └── config.toml                          # Custom Streamlit theme
├── .gitignore                               # Git ignore rules
└── README.md                                # Project documentation
```

---

## 💻 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/next-word-prediction-lstm-gru.git
   cd next-word-prediction-lstm-gru
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Launch the Streamlit Web App:**
   ```bash
   streamlit run app.py
   ```
   *(Optional)* Place `tuned_lstm_next_word.keras`, `tuned_gru_next_word.keras`, and `tokenizer_metadata.pkl` exported from the Colab notebook into the root folder to run live Keras inference, or run `app.py` directly using its built-in neural sequence engine!

---

## ☁️ Deploying on Streamlit Community Cloud

1. Push all files in this folder to a **GitHub Repository**.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and click **New app**.
3. Select your GitHub repository, branch (`main`), and set **Main file path** to `app.py`.
4. Click **Deploy!**
