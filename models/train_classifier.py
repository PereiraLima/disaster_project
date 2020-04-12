import sys
import pickle


import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

def load_data(database_filepath):
    """ Load db and table 'messages_cat' in df.
     Extract X and Y and return them"""
    engine = create_engine(database_filepath)
    df = pd.read_sql('SELECT * FROM messages_cat', engine)
    X = df.message
    Y = df.drop(columns=['genre', 'id', 'message', 'original'], axis=1)

    return X, Y

def tokenize(text):
    """ Create a list of tokens from a text"""

    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # import stop words: stop_words
    stop_words = stopwords.words("english")

    # create objec lemmatizer
    lemmatizer = WordNetLemmatizer()

    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens

def build_model():

    """ Build Pipeline using CountVectorizer and TfidfTransformer as feature extraction algorithm.
    Use Multinomial NB as classifier.
    Set different values to test for different hyperparameters
    Return model """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(MultinomialNB()))
    ])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'clf__estimator__alpha': [0.2, 0.6, 1]

    }

    cv = GridSearchCV(estimator=pipeline, param_grid=parameters)
    model_cv = cv.fit(X_train, Y_train)

    return model_cv

def evaluate_model(model, X_test, Y_test, category_names):

    Y_pred = model.predict(X_test)

    for i in range(len(category_names)):
        print('Accuracy Score column {}: \n {} \n'.format(category_names[i],
                                                          accuracy_score(Y_pred[:, i], Y_test.values[:, i])))


def save_model(model, model_filepath):

    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()