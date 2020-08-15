import sys

from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import re
import pickle

import nltk

nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score


def load_data(database_filepath):
    """Load the filepath and return the data separated into X and y datasets
    """
    name = 'sqlite:///' + database_filepath
    engine = create_engine(name)
    df = pd.read_sql_table('disaster_responses', con=engine)

    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    """Tokenize and transform input text.
    Return cleaned text
    """
    # take out all punctuation while tokenizing
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)

    # lemmatize as shown in the lesson
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """Return Grid Search model with pipeline and Classifier"""
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'vect__max_df': (0.5, 1.0),
        'tfidf__use_idf': (True, False)
    }

    cv = GridSearchCV(pipeline, parameters)
    return cv


def evaluate_model(model, X_test, y_test, category_names):
    """Print model results

    INPUT
    model: estimator-object
    X_test
    y_test
    category_names: list of category strings
    """
    # Get results and add them to a dataframe.
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=category_names))
    # results = pd.DataFrame(columns=['Category', 'f_score', 'precision', 'recall'])
    for i in range(len(category_names)):
        print("classification report for " + y_test.columns[i]
              , '\n', classification_report(y_test.values[:, i], y_pred[:, i])
              , '\n accuracy:', accuracy_score(y_test.values[:, i], y_pred[:, i]))


def save_model(model, model_filepath):
    """Save model as pickle file

    INPUT
    model: The model used for classification
    model_filepath: Path to the pkl file
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """Load the data, run the model and save model"""
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

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
