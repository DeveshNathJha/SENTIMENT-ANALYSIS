Twitter Sentiment Analysis using Machine Learning

Overview
This project is a Twitter Sentiment Analysis system that classifies tweets as positive or negative using Natural Language Processing (NLP) and Machine Learning (ML) techniques. The model is trained using a dataset of labeled tweets and employs Logistic Regression for classification.

Features
Data preprocessing: cleaning, stemming, and removing stopwords
TF-IDF vectorization for text representation
Model training using Logistic Regression
Performance evaluation on test data
Model saving and loading using pickle
Real-time sentiment prediction on new tweets

Dependencies
To run this project, install the required libraries using:
pip install numpy pandas scikit-learn nltk

Dataset
The dataset used is training.1600000.processed.noemoticon.csv, which contains labeled tweets with the following attributes:
target: Sentiment label (0 = Negative, 4 = Positive)
text: The tweet content

Data Preprocessing
Loading the dataset
Renaming columns for better understanding
Handling missing values
Stemming and stopword removal
TF-IDF Vectorization

Model Training & Evaluation
The data is split into training and testing sets (80%-20% ratio)
TF-IDF Vectorizer converts text into numerical form
Logistic Regression model is trained with 1000 iterations
Accuracy is calculated on both training and test data

Technologies Used
Python for scripting and model development
Pandas, NumPy for data manipulation and preprocessing
Scikit-learn for implementing machine learning models and evaluation metrics
Matplotlib, Seaborn for data visualization
NLTK for NLP processing and text cleaning

Results
The trained model classifies tweets into different sentiment categories with good accuracy. Performance metrics such as accuracy, precision, recall, and F1-score help evaluate its effectiveness.

Contact
For inquiries or collaborations, feel free to connect via GitHub or email.
Email: deveshjnv2002@gmail.com
