import sys
import pandas as pd
import numpy as np
import time
import pickle
import nltk
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD


def load_data(database_filepath):
    '''
    Fucntion to load the database 
    Input: Databased filepath
    Output: Returns  X, y and the columns names
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("disaster_response", con=engine)
    
    X = df["message"]
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    category_names = Y.columns
    print("category_names:", category_names)
    
    return X, Y, category_names


def tokenize(text):
    '''
    Function to tokenize the text messages
    Input: text
    Output: cleaned tokenized text as a list object
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    '''
    Function to build a model, pipeline or GridSearchCV
    Input: N/A
    Output: the model
    '''
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])
    
    parameters = {'tfidf__use_idf': (True, False), 
              'clf__estimator__n_estimators': [20, 50], 
              'clf__estimator__min_samples_split': [2, 4]} 

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv
        
        
def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Function to evaluate a model and print the classification report
    Input: model, X_test, y_test, catgegory_names
    Output: N/A
    '''
    y_pred = model.predict(X_test)
    print(classification_report(y_pred, Y_test.values, target_names=category_names))
    print('Accuracy: {}'.format(np.mean(Y_test.values == y_pred)))


def save_model(model, model_filepath):
    '''
    Function to save the model
    Input: model and the file path 
    Output: N/A
    '''
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()