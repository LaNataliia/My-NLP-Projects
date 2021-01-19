"""
Sentiment Classification
Bag-of-Words Representation
"""

import pandas as pd
import numpy as np
import sklearn
import re
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score


pd.set_option('display.max_columns', 99)
data = pd.read_csv('sentiment.csv')
data['sentiment'] = (data['sentiment'] == 'pos')


#%%
###############     DATA CLEANING AND BASIC STATISTICS     ###############


def clean_text(text):
    """Normalizing text to prevent model overfitting"""
    
    # remove all digits and symbols
    no_tags = re.sub(r"<[^>]*>", " ", text)
    letters_only = re.sub(r"[^a-zA-Z!?]", " ", no_tags)
    
    # lowercase and split to into separate words for further stopwords filtering
    tokens = nltk.tokenize.word_tokenize(letters_only.lower())
    words = [token for token in tokens if len(token)>1 or token in "!?"]
        
    # filtering out the stopwords
    stop_words = set(nltk.corpus.stopwords.words('english'))
    filtered_text = [word for word in words if not word in stop_words]
    
    # lemmatizing the remaining tokens
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(token) for token in filtered_text]
    
    # join everything back into one string
    normalized_text = (" ".join(lemmatized_words))
    
    return normalized_text    



# An option to use normalized or raw text

normalized_data = True
print()
if normalized_data:
    print('TEXT IS PRE-PROCESSED AND NORMALIZED')
    data['text'] = data['text'].apply(clean_text)
    print()
else:
    print('RAW TEXT IS USED, NO PRIOR NORMALIZATION')
    print()


#%%
###############     LOGISTIC REGRESSION WITH BAG-OF-WORDS REPRESENTATION     ###############



data_train, data_test = train_test_split(data, test_size=0.2, random_state=0)
y_train = data_train['sentiment']
y_test = data_test['sentiment']
    


def logreg_with_bow(data_train, data_test, y_train, y_test, bow_type):
    """Universal logistic regression with bag of words representation
        for single words, bigrams or tf-idf (the choice passed as a parameter);
        includes model evaluation"""
    
    if bow_type == 'unigrams':
        vectorizer = CountVectorizer()
    if bow_type == 'bigrams':
        vectorizer = CountVectorizer(ngram_range = (2,2))
    if bow_type == 'tf-idf':
        vectorizer = TfidfVectorizer()
    
    vectorizer.fit(data_train['text'])
    X_train = vectorizer.transform(data_train['text'])
    X_test = vectorizer.transform(data_test['text'])

    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Model evaluation
    print ('Bag-of-Words model used with ', bow_type.upper())
    print("Accuracy: ", model.score(X_test, y_test))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print()
    print("Cross-validation: ", cross_val_score(model, X_test, y_test, cv=10))
    print()
    
    coefficients = model.coef_.squeeze()
    feature_names = vectorizer.get_feature_names()
    word_importances = list(zip(coefficients, feature_names))
    word_importances.sort()
    print(bow_type.upper(),"WITH THE MOST NEGATIVE COEFFS (descending):")
    for i in range (15):
        print(word_importances[i][1])
    print()
    print(bow_type.upper(), "WITH THE MOST POSITIVE COEFFS (ascending):")
    for y in range ((len(word_importances)-15), len(word_importances)):
        print(word_importances[y][1])
    print()
    
    
    
# Logistic regression for words/ bi-grams/ tf-idf

bow_types = ['unigrams', 'bigrams', 'tf-idf']
for bow in bow_types:
    logreg_with_bow(data_train, data_test, y_train, y_test, bow)

    

