# import libraries
import sys
import pandas as pd
import numpy as np
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
nltk.download(['punkt', 'wordnet', 'stopwords'])
import pickle


def load_data(database_filepath):
    """
    Loads data from SQLite database.
    
    Parameters:
    database_filepath: Filepath to the database
    
    Returns:
    X: Features
    Y: Target
    """
    # load data from database 
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("disaster_messages", con=engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    return X,Y


def tokenize(text):
    """
    Normalize, Tokenizes, Stems, and lemmatizes text.
    
    Parameters:
    text: Text to be tokenized
    
    Returns:
    clean_tokens: Returns cleaned tokens 
    """
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    stop_words = stopwords.words("english")

    # tokenize
    tokens = word_tokenize(text)
    
    # Stemming
    stems = [PorterStemmer().stem(t) for t in tokens]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
       
    clean_tokens=[]
    for tok in stems:
        if tok not in stop_words:
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model():
    """
    Builds classifier and tunes model using Linear SVC.
    
    Returns:
    pipeline_svc: Classifier 
    """    
    pipeline_svc = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', OneVsRestClassifier(LinearSVC()))
    ])

    return pipeline_svc


def evaluate_model(model, X_test, Y_test):
    """
    Evaluates the performance of model and returns classification report, accuracy and f1 score. 
    
    Parameters:
    model: classifier
    X_test: test dataset
    Y_test: labels for test data in X_test
    
    Returns:
    Classification report for each column
    Mean Accuracy
    Mean F1 c=score
    """
    y_pred_svc = model.predict(X_test)
       
    acc = []
    f1 = []
    for index, column in enumerate(Y_test):
        print(column, classification_report(Y_test[column], y_pred_svc[:, index]))
        f1.append(f1_score(Y_test[column], y_pred_svc[:, index], average = 'macro'))
        acc.append(accuracy_score(Y_test[column], y_pred_svc[:, index]))

    print("-------------------------------------------------------")    
    print("Mean accuracy score: {:.4f}".format(np.mean(acc)))
    print("Mean f1-score (macro): {:.4f}".format(np.mean(f1)))
        


def save_model(model, model_filepath):
    """ Exports the model as a pickle file."""
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()