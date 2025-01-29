# SentimentAnalysis-NLP
Libraries Used and Their Functions:

pandas: Facilitates data manipulation and analysis using DataFrames.

numpy: Supports numerical operations with arrays and matrices.

matplotlib.pyplot: Enables the creation of static visualizations.

seaborn: Extends matplotlib to provide high-level statistical graphics.

wordcloud: Generates graphical representations of text data.

nltk (Natural Language Toolkit): Provides tools for text processing and NLP tasks such as:

nltk.corpus: Grants access to various text corpora.

nltk.tokenize: Implements methods for splitting text into words or sentences.

nltk.stem: Includes stemming and lemmatization algorithms for text normalization.

nltk.tokenize.toktok: Offers a fast tokenizer for NLP applications.

re (Regular Expressions): Supports pattern matching in strings.

scikit-learn: Provides machine learning tools, including:

sklearn.model_selection: Splits data into training and testing sets.

sklearn.feature_extraction.text: Converts text into numerical representations (e.g., TF-IDF, CountVectorizer).

sklearn.preprocessing: Includes data preprocessing techniques such as label binarization.

sklearn.linear_model: Implements linear classification models, including Logistic Regression.

sklearn.metrics: Offers evaluation metrics for machine learning models.

sklearn.naive_bayes: Implements Naive Bayes classifiers.

Project Overview:

This project focuses on sentiment analysis, word cloud generation, and the implementation of LSTM and LDA models.

Data Exploration and Preprocessing:

Utilizes pandas for data manipulation, seaborn and matplotlib for visualization, nltk for text processing, and scikit-learn for machine learning tasks.

Generates a sentiment distribution plot using sns.countplot.

Analyzes word length distribution for positive and negative reviews.

Creates word clouds to visualize frequently occurring words in positive and negative sentiments.

Implements functions to clean review text by removing HTML tags, special characters, and stop words.

Splits the dataset into training and testing sets.

Feature Extraction:

Converts text data into numerical representations using:

CountVectorizer (Bag-of-Words)

TfidfVectorizer (TF-IDF)

Sentiment Classification Models:

Trains two Logistic Regression models:

One using Bag-of-Words features.

Another using TF-IDF features.

Evaluates model performance using classification reports.

LSTM Model:

Implements an LSTM model for sentiment classification.

Trains the model on processed text data, potentially using word embeddings.

Visualizes training and validation accuracy during model training.

LDA Topic Modeling:

Preprocesses text data for topic modeling by removing stop words and applying stemming.

Generates a dictionary mapping words to unique identifiers.

Converts processed reviews into a bag-of-words format.

Constructs an LDA model to identify hidden topics within reviews.

Displays the most relevant words for each discovered topic.

https://colab.research.google.com/drive/1zypCQpGN3c402jx_o2OxjdKwIRHjZfRs?usp=sharing#scrollTo=Dn14DYT2flLC
