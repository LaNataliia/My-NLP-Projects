"""
Nataliia Zolotukhina

Sarcasm Detection in Twitter
and the Correlation Between Sarcasm and Sentiment
"""


#%%
###############     IMPORTING MODULES AND LOADING DATA     ###############



import pandas as pd
import numpy as np
import re
import nltk
import spacy
from nltk.tokenize import TweetTokenizer
from nltk.corpus import sentiwordnet as swn
from textblob import TextBlob

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


pd.set_option('display.max_columns', 99)
tk = TweetTokenizer()
nlp = spacy.load("en_core_web_md")
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
stopwords = set(nltk.corpus.stopwords.words('english'))
sarcasm_markers = ['sarcasm', 'sarcastic']


sarcasmdata = pd.read_excel('sarcasmdata.xlsx')
sarcasmdata = sarcasmdata.sample(frac=1, random_state=42).reset_index(drop=True)    # shuffle the dataset
sarcasmdata.columns = ['tweet', 'class']
sarcasmdata.drop_duplicates(subset ="tweet", keep = 'first', inplace = True)    # remove duplicates
sarcasmdata['is_sarcastic'] = (sarcasmdata['class'] == 'sarcasm')


#%%
###############     BASIC STATISTICS AND DATA CLEANING     ###############



def get_basic_stats():
    """Count of total and unique values per column,
        count of tweets per class, the plot of tweets distribution per class"""
    
    print('Basic statistics for the dataset:')
    print(sarcasmdata.describe().T)
    print()
    
    print('Amount of tweets per class:')
    print(sarcasmdata['class'].value_counts())
    print()
    
    splot = sns.countplot(sarcasmdata['class'])
    plt.title('Distribution of Tweets')
    for p in splot.patches:
        splot.annotate('{}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+50))

get_basic_stats()

 

def clean_tweet(tweet):
    """Removing noise, normalizing text and preparing it for the further processing"""
    
    # Special characters
    tweet = re.sub(r"\x89Ûª(?=s)", "'", tweet)
    tweet = re.sub(r"\x89Û\x9d", "", tweet)
    tweet = re.sub(r"\x89Û.", "", tweet)
    tweet = re.sub(r"donå«t", "don't", tweet)
    tweet = re.sub(r"å.", "", tweet)
    tweet = re.sub(r"JapÌ_n", "Japan", tweet)    
    tweet = re.sub(r"Ì©", "e", tweet)
    tweet = re.sub(r"SuruÌ¤", "Suruc", tweet)
    tweet = re.sub(r"вЂ.", "'", tweet)

    # usernames mentions
    tweet = re.sub(r'@[A-Za-z0-9_]*', '', tweet)
    
    # character entity references
    tweet = re.sub(r"&amp;", "&", tweet)
    
    # html tags
    tweet = re.sub(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});', '', tweet)
    
    # Urls
    tweet = re.sub(r'https?:\S+(?=\s|$)', '', tweet)
    
    # Punctuations and special characters
    punctuation_to_cut='!#$%&\()*+,-./:;<=>?@[\\]^_{|}~'
    tweet = re.sub('[%s]' % re.escape(punctuation_to_cut),'',tweet)
    
    # Non-English letters
    tweet = re.sub(r"[^\u0000-\u007F]+", "", tweet)
       
    # contractions = ["'re", "'s", "n't", "'d", "'ll", "'ve"]
    tokens = tk.tokenize(tweet)
    tokens = [word for word in tokens if word.lower() not in stopwords]
    tokens = [word for word in tokens if word.lower() not in sarcasm_markers]
    tweet = ' '.join(tokens)
    
    tweet = re.sub(r'\s[\'\"]', '', tweet) 
    tweet = re.sub(r'[\'\"]\s', '', tweet) 

    return tweet

sarcasmdata['cleaned_tweet'] = sarcasmdata['tweet'].apply(clean_tweet)


# get_basic_stats()
sarcasmdata.drop_duplicates(subset ="cleaned_tweet", keep = 'first', inplace = True)
# get_basic_stats()



def get_word_clouds(text, title):
    """Visualizing a text string as a WordClous"""
    plt.figure(figsize=(10,5))
    plt.imshow(WordCloud(background_color='white').generate(text))
    plt.title(title,fontsize=25)
    plt.axis('off')
    print()
        
regular_tweets = ' '.join(sarcasmdata.cleaned_tweet[sarcasmdata['is_sarcastic'] == False])
sarcastic_tweets = ' '.join(sarcasmdata.cleaned_tweet[sarcasmdata['is_sarcastic'] == True])
get_word_clouds(regular_tweets, 'Regular Tweets')
get_word_clouds(sarcastic_tweets, 'Sarcastic Tweets')


#%%
###############     BASIC FEATURES     ###############



def get_spacy_tokens(cleaned_tweet):
    """Tokenizing a string in spaCy"""
    
    text = nlp(cleaned_tweet)
    tokens = []
    for token in text:
        tokens.append(token.text)
    return tokens

#sarcasmdata['spacy_tokens'] = sarcasmdata['cleaned_tweet'].apply(get_spacy_tokens)

sarcasmdata['nltk_tokens'] = sarcasmdata['cleaned_tweet'].apply(tk.tokenize)
sarcasmdata['nltk_pos'] = sarcasmdata['nltk_tokens'].apply(nltk.pos_tag)



def get_hashtags(tweet):
    """Extract hashtags from a raw tweet; #sarcasm and #sarcastic excluded"""
    tokens = tk.tokenize(tweet)
    hashtags = [token for token in tokens if token[0] == '#' and token.lower()[1:] not in sarcasm_markers]
    return hashtags

sarcasmdata['hashtags'] = sarcasmdata['tweet'].apply(get_hashtags)



def get_interjections(cleaned_tweet):
    """Extracting interjections from a text string"""
    
    pattern = """
       (?<=\s)y+e+a*h*(?=\s)
    |  (?<=\s)y+a+y+(?=\s)
    |  (?<=\s)l+o+l+(?=\s)
    |  (?<=\s)l+m+a+o+(?=\s)
    |  (?<=\s)o+h+(?=\s)
    |  (?<=\s)w+o+w+(?=\s)
    |  (?<=\s)a+w+(?=\s)
    |  (?<=\s)o+p+s+(?=\s)
    |  (?<=\s)w+o+h+o+(?=\s)
    |  (?<=\s)w+t+f+(?=\s)
    |  (?<=\s)a+h+a+(?=\s)
    |  (?<=\s)y+u+p+(?=\s)
    |  (?<=\s)y+e+a+p+(?=\s)
    |  (?<=\s)a*h+a+h+a+(?=\s)
    |  (?<=\s)h+e+h+e+(?=\s)
    """
    
    text = ' ' + cleaned_tweet + ' '
    interjections = re.findall(pattern, text.lower(), re.VERBOSE)
    return interjections

sarcasmdata['interjections'] = sarcasmdata['cleaned_tweet'].apply(get_interjections)



def check_capitalized_words(tokenized_tweet):
    """Checking the presence of capitalized words longer then 3 characters in a tokenized sentence"""
    capitalized_words = []
    for token in tokenized_tweet:
        if token.isupper() and len(token)>3:
            capitalized_words.append(token)
    # return len(capitalized_words)
    if capitalized_words:
        return True
    else:
        return False
    

    
def compute_basic_features():
    """Implementing basic features for training a Logistic Regression model"""
    
    # Numeric features
    sarcasmdata['n_nltk_tokens'] = sarcasmdata['nltk_tokens'].apply(len)
    sarcasmdata['n_hashtags'] = sarcasmdata['hashtags'].apply(len)
    sarcasmdata['n_!|?'] = sarcasmdata['tweet'].apply(
        lambda x: len(re.findall(r'\?|\!', x)))
    
    # Boolean features
    sarcasmdata['mentions'] = sarcasmdata['tweet'].apply(
        lambda x: True if re.findall(r'@[A-Za-z0-9_]*', x) else False)
    sarcasmdata['has_interjections'] = sarcasmdata['interjections'].apply(
        lambda x: True if x else False)
    sarcasmdata['tripled_letters'] = sarcasmdata['cleaned_tweet'].apply(
        lambda x: True if re.findall(r'(\w)\1\1', x) else False)
    sarcasmdata['capitalized_tokens'] = sarcasmdata['nltk_tokens'].apply(check_capitalized_words)

compute_basic_features()


# Cross check - How relevant the features are?
print()
print(sarcasmdata.groupby('class')[['n_nltk_tokens', 'n_hashtags', 'n_!|?']].mean())
print()
print(sarcasmdata.groupby('class')[['mentions', 'has_interjections', 'tripled_letters', 'capitalized_tokens']].mean())
print()

  

#%%
###############     BASELINE SOLUTION     ###############



def get_accuracy_score (system, gold):
    """Computing accuracy by comparing system output with golden tags"""
    return (system == gold). mean()



def get_baseline_solution():
    """Implementing a baseline solution:
        randomly assigning sarcastic tags keeping the class ballance"""
        
    prop_sarcastic = sarcasmdata['is_sarcastic'].mean()
    prop_regular = 1 - prop_sarcastic
    sarcasmdata['baseline'] = np.random.choice([True, False], size=len(sarcasmdata),
                                           p=[prop_sarcastic, prop_regular])
    print ('Baseline accuracy: ', get_accuracy_score(sarcasmdata['baseline'], sarcasmdata['is_sarcastic']))
    print()

get_baseline_solution()


#%%
###############     LOGISTIC REGRESSION     ###############



def logistic_regression(df, X, y):
    """Training and testing the Logistic Regression model"""

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)   # for overfitting check
    
    print("LOGISTIC REGRESSION SUCCESSFULLY IMPLEMENTED!")
    print()
    print("Testing accuracy:  ", accuracy_score(y_test, y_pred))
    print("Training accuracy: ", accuracy_score(y_train, y_train_pred))
    # print("Cross-validation:  ", cross_val_score(model, X_test, y_test, cv=10))
    print()
    
    print()
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print()
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    print()
    print("Model coefficients:")
    for idx, col_name in enumerate(X_train.columns):
        print("{:<25} --> {:>8.3f}".format(col_name, model.coef_[0][idx]))
    print()



y = sarcasmdata['is_sarcastic']
X1 = sarcasmdata[['n_nltk_tokens', 'n_hashtags', 'n_!|?',
                  'mentions', 'has_interjections', 'tripled_letters', 'capitalized_tokens']]


# Logistic Regression with first 7 (basic) features
print()
print('###########################################################')
print('LOGISTIC REGRESSION WITH BASIC FEATURES:')
print()
logistic_regression(sarcasmdata, X1, y)
print()


#%%
###############     SENTIMENT AS A FEATURE     ###############



def get_textblob_sentiment():
    """Getting sentiment and subjectivity scores from TextBlob"""
    
    sarcasmdata['textblob_sentiment'] = sarcasmdata['cleaned_tweet'].apply(
        lambda x: TextBlob(x).sentiment[0])
    sarcasmdata['textblob_subjuctivity'] = sarcasmdata['cleaned_tweet'].apply(
        lambda x: TextBlob(x).sentiment[1])

get_textblob_sentiment()


print()
print(sarcasmdata.groupby('class')[['textblob_sentiment', 'textblob_subjuctivity']].mean())
print()


X2 = sarcasmdata[['n_nltk_tokens', 'n_hashtags', 'n_!|?',
                  'mentions', 'has_interjections', 'tripled_letters', 'capitalized_tokens',
                  'textblob_sentiment', 'textblob_subjuctivity']]

print()
print('###########################################################')
print('BASIC FEATURES + SENTIMENT:')
print()
logistic_regression(sarcasmdata, X2, y)
print()


#%%
###############     ADVANCED LEXICON FEATURES     ###############



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

positive_lexicon, negative_lexicon = get_lexicon()



def get_swn_scores(tagged_tweet):
    """Collecting positive and negative scores for each token in a tweet"""
    
    synset_tags = {"JJ": "a", "JJR": "a", "JJS": "a",
                   "NN": "n", "NNS": "n", "NNP": "n", "NNPS": "n",
                   "RB": "r", "RBR": "r", "RBS": "r",
                   "VB": "v", "VBD": "v", "VBG": "v", "VBN": "v", "VBP": "v", "VBZ": "v"}
    
    swn_scores = []
    
    for token_pos_pair in tagged_tweet:
        lemma = lemmatizer.lemmatize(token_pos_pair[0].lower())
        tag = token_pos_pair[1]
        
        if tag in synset_tags.keys():
            synsets = list(swn.senti_synsets(lemma, synset_tags[tag]))
        
            if synsets:     # verifying it's not an empty set
                first_synset = synsets[0]   # only considering the first synset for each word
                swn_scores.append((lemma, tag, first_synset.pos_score(), first_synset.neg_score()*(-1)))
            else:
                swn_scores.append((lemma, tag, 0.0, -0.0))
        else:
            swn_scores.append((lemma, tag, 0.0, -0.0))    
                
    return swn_scores
    
sarcasmdata['swn_scores'] = sarcasmdata['nltk_pos'].apply(get_swn_scores)



def check_polarity_change_swn(swn_scores_list):
    """Checking whether sentiment scores change polarity in the beginning and in the end of a tweet"""
    
    half_length = len(swn_scores_list)//2
    tweet_beginning_score = 0
    tweet_end_score = 0
    
    for i in range (0, half_length):
        tweet_beginning_score = tweet_beginning_score + swn_scores_list[i][2] + swn_scores_list[i][3]
    for j in range (half_length, len(swn_scores_list)):
        tweet_end_score = tweet_end_score + swn_scores_list[j][2] + swn_scores_list[j][3]
    
    if tweet_beginning_score * tweet_end_score < 0:
        return True
    else:
        return False



def check_polarity_change_lexicon(tokens_list):
    """Checking the number of positive and negative words in the beginning and in the end of a tweet.
        If one half has more positive words and the other half has more negative words,
        there is a change in polarity."""
    
    length = len(tokens_list)
    half_length = len(tokens_list)//2
    n_pos_beginning = 0
    n_pos_end = 0
    n_neg_beginning = 0
    n_neg_end = 0
    
    for i in range (0, half_length):
        if tokens_list[i].lower() in positive_lexicon:
            n_pos_beginning += 1
        if tokens_list[i].lower() in negative_lexicon:
            n_neg_beginning += 1
            
    for j in range (half_length, length):
        if tokens_list[j].lower() in positive_lexicon:
            n_pos_end += 1
        if tokens_list[j].lower() in negative_lexicon:
            n_neg_end += 1
    
    if (n_pos_beginning - n_neg_beginning) * (n_pos_end - n_neg_end) < 0:
        return True
    else:
        return False



def get_max_mean_stdev(list_of_swn_scores):
    """Computing the maximum, the average, and the standard deviation values
        of the positive and negative scores obtained from SentiWordNet"""
        
    pos_scores = [item[2] for item in list_of_swn_scores]
    neg_scores = [item[3] for item in list_of_swn_scores]
    all_scores = pos_scores + neg_scores
       
    if all_scores:
        maximum = np.max(all_scores)
        average = np.mean(all_scores)
        st_deviation = np.std(all_scores)
    else:
        maximum = 0
        average = 0
        st_deviation = 0
        
    return maximum, average, st_deviation
        
# sarcasmdata['swn_max'] = sarcasmdata['swn_scores'].apply(lambda x: get_max_mean_stdev(x)[0])
# sarcasmdata['swn_mean'] = sarcasmdata['swn_scores'].apply(lambda x: get_max_mean_stdev(x)[1])



def compute_advanced_features():
    """Implementing advanced features for training a Logistic Regression model"""
    
    # Numeric features
    sarcasmdata['n_pos_words'] = sarcasmdata['nltk_tokens'].apply(
        lambda x: len([token for token in x if token.lower() in positive_lexicon]))
    sarcasmdata['n_neg_words'] = sarcasmdata['nltk_tokens'].apply(
        lambda x: len([token for token in x if token.lower() in negative_lexicon]))
    sarcasmdata['swn_stdev'] = sarcasmdata['swn_scores'].apply(lambda x: get_max_mean_stdev(x)[2])
    sarcasmdata['avg_max_diff'] = sarcasmdata['swn_scores'].apply(
        lambda x: get_max_mean_stdev(x)[0] - get_max_mean_stdev(x)[1])
    
    # Boolean features
    sarcasmdata['polarity_change_swn'] = sarcasmdata['swn_scores'].apply(check_polarity_change_swn)
    sarcasmdata['polarity_change_lexicon'] = sarcasmdata['nltk_tokens'].apply(check_polarity_change_lexicon)
    sarcasmdata['neg_avg/pos_max'] = sarcasmdata['swn_scores'].apply(
        lambda x: True if get_max_mean_stdev(x)[0] > 0 and get_max_mean_stdev(x)[1] < 0
        else False)

compute_advanced_features()


# Cross check - How relevant the features are?
print()
print(sarcasmdata.groupby('class')[['n_pos_words', 'n_neg_words', 'swn_stdev', 'avg_max_diff']].mean())
print()
print(sarcasmdata.groupby('class')[['polarity_change_swn', 'polarity_change_lexicon', 'neg_avg/pos_max']].mean())
print()


X3 = sarcasmdata[['n_nltk_tokens', 'n_hashtags', 'n_!|?',
                  'mentions', 'has_interjections', 'tripled_letters', 'capitalized_tokens',
                  'textblob_sentiment', 'textblob_subjuctivity',
                  'n_pos_words', 'n_neg_words', 'swn_stdev', 'avg_max_diff',
                  'polarity_change_swn', 'polarity_change_lexicon', 'neg_avg/pos_max']]

print()
print('###########################################################')
print('BASIC FEATURES + SENTIMENT + ADVANCED FEATURES:')
print()
logistic_regression(sarcasmdata, X3, y)
print()   


