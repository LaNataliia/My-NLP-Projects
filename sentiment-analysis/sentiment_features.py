
"""
Sentiment Classification
Hand-Crafted Features
"""



import pandas as pd
import numpy as np
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import sentiwordnet as swn

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


pd.set_option('display.max_columns', 99)
data = pd.read_csv('sentiment.csv')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


#%%
###############     DATA PRE-PROCESSING AND BASIC STATISTICS     ###############



def prepare_dataset(df):
    """Preparing the DataFrame for further processing:
        modification of the "sentiment" column, tokenization, POS tagging,
        stopwords removal, tokens/sentences count""" 
    df['sentiment'] = df['sentiment'] == 'pos'
    df['n_characters'] = df['text'].apply(len)
    df['tokens'] = df['text'].apply(nltk.tokenize.word_tokenize)
    df['n_tokens'] = df['tokens'].apply(len)
    df['sentences'] = df['text'].apply(nltk.tokenize.sent_tokenize)
    df['n_sentences'] = df['sentences'].apply(len)
    df['POS tags'] = df['tokens'].apply(nltk.pos_tag)
    df['filtered_review'] = df['POS tags'].apply(
        lambda x: [token_tag for token_tag in x if not lemmatizer.lemmatize(token_tag[0]) in stop_words]
      )
    df['n_filtered_tokens'] = df['filtered_review'].apply(len)

prepare_dataset(data)



def get_basic_stats(df):
    """Computing basic statistics of the dataset,
        plotting histogram of the "rating" column"""
    
    print()
    print("Basic statistics of the dataset:")
    print(df[["rating", "sentiment", "n_characters", "n_tokens", "n_sentences"]].describe())
    print()
    print("Average number of words per review: ", df['n_tokens'].mean())
    print("Average number of sentences per review: ", df['n_sentences'].mean())
    print()
    print("Average number of words and sentences in positive(True) and negative(False) reviews:")
    print(df.groupby('sentiment')[['n_tokens', 'n_sentences']].mean())
    print()

get_basic_stats(data)


#%%
###############     FEATURES IMPLEMENTATION     ###############



def get_lexicon():
    """Getting the lists of positive and negative words,
        based on the Opinion Lexicon by Liu & Hu (2004)"""
    positive_lexicon = []
    negative_lexicon = []
    with open("positive-words.txt", "r") as f_pos:
        for line in f_pos.readlines():
            positive_lexicon.append(line.strip())
    with open("negative-words.txt", "r") as f_neg:
        for line in f_neg.readlines():
            negative_lexicon.append(line.strip())
    return (positive_lexicon, negative_lexicon)


def compute_features(df):
    """Computing feature relevant for sentiment classification"""
    
    positive_lexicon, negative_lexicon = get_lexicon()
    
    df['F1: n_pos_words'] = df['tokens'].apply(
        lambda x: len([token for token in x if lemmatizer.lemmatize(token.lower()) in positive_lexicon]))
    df['F2: n_neg_words'] = df['tokens'].apply(
        lambda x: len([token for token in x if lemmatizer.lemmatize(token.lower()) in negative_lexicon]))
    df['F3: adj_ratio'] = df['filtered_review'].apply(
        lambda x: len([token_tag[0] for token_tag in x if token_tag[1] in ['JJ', 'JJR', 'JJS']])/len(x))
    df['F4: avg_sent_length'] = df['n_tokens']/df['n_sentences']
    df['F5: filtered_text_ratio'] = df['n_filtered_tokens']/df['n_tokens']

compute_features(data)


#%%
###############     BASELINE SOLUTIONS     ###############



def get_baseline_predictions(df):
    """Establishing 3 baselines:
        everything positive/ everything negative/ random"""
    df["Baseline Pos"] = True
    df["Baseline Neg"] = False
    prop_positive = df['sentiment'].mean()
    prop_negative = 1 - prop_positive
    df['Baseline Rand'] = np.random.choice([True, False], size=len(data),
                                           p=[prop_positive, prop_negative])

def get_accuracy_score (system, gold):
    """Compares system outputs with the gold labels,
        returns the accuracy score"""
    return (system == gold). mean()

def get_accuracies(list_of_columns):
    """Computes accuracy score for specific columns"""
    print ('Baseline accuracies:')
    for column in list_of_columns:
        acc = get_accuracy_score(data[column], data['sentiment'])
        print (column,': ', acc)
    print()

get_baseline_predictions(data)
get_accuracies(['Baseline Pos', 'Baseline Neg', 'Baseline Rand'])
 

#%%
###############     LOGISTIC REGRESSION     ###############

    

X1 = data[['F1: n_pos_words', 'F2: n_neg_words',
          'F3: adj_ratio', 'F4: avg_sent_length',
          'F5: filtered_text_ratio']]
y = data['sentiment']


def logistic_regression(df, X, y):
    """Training the LogReg model on the given features,
    testing it and evaluating its performance"""
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("LOGISTIC REGRESSION SUCCESSFULLY IMPLEMENTED!")
    print()
    print("Accuracy: ", model.score(X_test, y_test))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print()
    print("Model coefficients:")
    for idx, col_name in enumerate(X_train.columns):
        print("{:<25} --> {:>8.3f}".format(col_name, model.coef_[0][idx]))
    print()

logistic_regression(data, X1, y)
    

# # Cross check: - How relevant the chosen features are?
# print(data.groupby('sentiment')[['F1: n_pos_words', 'F2: n_neg_words']].mean())
# print(data.groupby('sentiment')[['F3: adj_ratio', 'F4: avg_sent_length', 'F5: filtered_text_ratio']].mean())
                      

#%%
###############     ADDITIONAL FEATURE - SentiWordNet     ###############


def get_swn_score(tagged_text):
    """Use SWN, count scores for nouns/verbs/adjectives/adverbs only;
    add positive scores and subtract negative scores from the general score"""
    
    synset_tags = {"JJ": "a", "JJR": "a", "JJS": "a",
                   "NN": "n", "NNS": "n", "NNP": "n", "NNPS": "n",
                   "RB": "r", "RBR": "r", "RBS": "r",
                   "VB": "v", "VBD": "v", "VBG": "v", "VBN": "v", "VBP": "v", "VBZ": "v"}
    
    score = 0
    for pos_pair in tagged_text:
        lemma = lemmatizer.lemmatize(pos_pair[0].lower())
        tag = pos_pair[1]
        if tag in synset_tags.keys():
            synsets = list(swn.senti_synsets(lemma, synset_tags[tag]))
        
            if synsets: # verifying it's not an empty set
                first_synset = synsets[0] # only considering the first option for each word
                if first_synset.pos_score() != 0:
                    score = score + first_synset.pos_score()
                else:
                    score = score - first_synset.neg_score()
    return score

data['F6: SWN Score'] = data['POS tags'].apply(get_swn_score)


# # Choosing a threshold for a rule-based classification
# print(data.groupby('sentiment')[['F6: SWN Score']].mean())


# Applying rule-based classification and checking its accuracy
data['SWN Prediction'] = data['F6: SWN Score'].apply(lambda x: True if x>=5 else False)
print('SWN rule-based aclassification accuracy: ', get_accuracy_score (data['SWN Prediction'], data['sentiment']))
print()


# Logistic regression with the additional feature

X2 = data[['F1: n_pos_words', 'F2: n_neg_words',
          'F3: adj_ratio', 'F4: avg_sent_length',
          'F5: filtered_text_ratio', 'F6: SWN Score']]
y = data['sentiment']

logistic_regression(data, X2, y)

    
    
    
    
    
    
    
    
    
    
    
    
    