
# Comparative Analysis of Deep Learning Models for Sentiment Analysis on Yelp Reviews

This repository contains the code, results, and documentation for the final project in the Advanced Deep Learning (AIGC 5500) course. The goal of this project is to compare the performance of two deep learning models—LSTM and DistilBERT—on a multi-class sentiment analysis task using Yelp restaurant and hotel reviews.

## Project Overview

Understanding customer sentiment is crucial for businesses, especially in the hospitality sector. This project classifies Yelp reviews into positive, neutral, or negative sentiments using two different deep learning approaches:

- LSTM (Long Short-Term Memory): A recurrent neural network that captures sequential dependencies in text.
- DistilBERT: A transformer-based model that encodes deep contextual information from text efficiently.

## Dataset

The dataset includes Yelp reviews along with their star ratings. Reviews are categorized into sentiment classes as follows:

- Positive: 4 or 5 stars  
- Neutral: 3 stars  
- Negative: 1 or 2 stars

### Preprocessing Steps

- HTML tag, punctuation, and number removal  
- Contraction expansion (e.g., "doesn't" to "does not")  
- Lemmatization using NLTK WordNet  
- Stopword removal, while retaining sentiment-relevant words  
- Class balancing using upsampling  
- Tokenization and sequence preparation:
  - For LSTM: Tokenization and padding, followed by word embeddings
  - For DistilBERT: Transformer-specific tokenization using pre-trained tokenizer

## Model Architectures

### LSTM Model

- Embedding Layer  
- Bidirectional LSTM Layer  
- Dropout Layer  
- Dense Output Layer with Softmax Activation

### DistilBERT Model

- Pre-trained DistilBERT Encoder Layer  
- Dense Layer with ReLU Activation  
- Dropout Layer  
- Dense Output Layer with Softmax Activation

## Training Details

- Optimizer: Adam  
- Loss Function: Sparse Categorical Crossentropy  
- Epochs: 3  
- Evaluation: Accuracy, Confusion Matrix, Classification Report  
- Interpretability: LIME used for explaining model predictions

## Results

| Metric                     | LSTM        | DistilBERT   |
|---------------------------|-------------|--------------|
| Accuracy                  | Moderate    | Higher       |
| Short Review Performance  | Good        | Excellent    |
| Long Review Performance   | Lower       | Strong       |
| Interpretability (LIME)   | Moderate    | High         |
| Training Time             | Fast        | Slower       |
| Resource Efficiency       | High        | Lower        |

## Key Findings

- DistilBERT achieved higher accuracy and performed better on longer and more complex reviews.
- LSTM offered faster training and required fewer resources, making it suitable for low-resource environments.
- LIME helped identify which words and phrases influenced the models' decisions, enhancing interpretability.

## Future Work

- Experiment with domain-specific embeddings like GloVe trained on Yelp data.
- Use ensemble techniques to combine the strengths of both models.
- Explore zero-shot or few-shot learning with newer transformer models.

## Team Contribution

| Team Member | Responsibilities |
|-------------|------------------|
| Antra       | Data preprocessing, LSTM implementation, LSTM performance analysis |
| Kartik      | DistilBERT implementation, model comparison, LIME interpretability |
| Pushkar     | Report writing, visualizations, presentation design |

## Repository Structure

```
sentiment-analysis-yelp/
│
├── data/                   # Yelp review dataset
├── models/                 # LSTM and DistilBERT model architectures
├── notebooks/              # Jupyter notebooks for each phase of the project
├── results/                # Graphs, evaluation outputs, confusion matrices
├── final_Project_report.pdf # Full project report
└── README.md               # Project summary and documentation
```

## License

This project is intended for academic and research purposes under the MIT License.
