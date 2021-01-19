"""
Word Sense Disambiguation
"""


import pandas as pd
import numpy as np
import nltk
import re

from nltk.corpus import wordnet
from nltk.stem import porter
from nltk.util import bigrams

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


pd.set_option('display.max_columns', 99)
stop_words = set(nltk.corpus.stopwords.words('english'))
stemmer = porter.PorterStemmer()


#%%
###############     DATA PRE-PROCESSING     ###############



# Data Loading

data = pd.read_csv('semcor.csv')
del data['sentence_id']
del data['target_index']
data.fillna('', inplace=True)
data['full_context'] = data['context_before'] + ' ' + data['target_word'] + ' ' + data['context_after']


# Word2Vec Data

def get_words_and_vectors():
    """Getting data from nltk sample of word2vec embeddings:
        a list of words,
        a list of corresponding vectors,
        a dictionary with word-vector mapping"""
    path_to_word2vec_sample = nltk.data.find('models/word2vec_sample/pruned.word2vec.txt')
    words = []
    vectors = []
    word2vec = {}
    
    with open(path_to_word2vec_sample) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            line = line.split()
            word = line[0]
            vector = line[1:]
            vector = [float(s) for s in vector]
        
            words.append(word)
            vectors.append(vector)
            word2vec[word] = vector
    return (words, vectors, word2vec)

words, vectors, word2vec = get_words_and_vectors()


def norm(vector):
    """Compute the length of a vector"""
    return np.linalg.norm(vector)

def dot(vector1, vector2):
    """Compute the dot product of two vectors"""
    return sum([i * j for i, j in zip(vector1, vector2)])

def cosine(vector1, vector2):
    """Compute the cosine of two vectors"""
    return dot(vector1, vector2) / (norm(vector1) * norm(vector2))


#%%
###############     BASELINE SOLUTIONs     ###############


    
def get_most_frequent_synset(word):
    """Given a word, the function returns the name of its most frequent synset
        (the first one listed in WordNet)"""
    frequent_synset = str(wordnet.synsets(word)[0])
    frequent_synset_name = re.findall(r'\w+\.\w\.\d+', frequent_synset)[0]
    return frequent_synset_name

data['most_frequent_synset'] = data['target_word'].apply(get_most_frequent_synset)

# Baseline1 : the correct synset is the most frequent synset
data['baseline_on_frequency'] = (data['synset'] == data['most_frequent_synset'])



# Baseline 2 - random classification, with corresponding proportion (1/3 True - 2/3 False)
prop_positive = data['synset_is_correct'].mean()
prop_negative = 1 - prop_positive
data['baseline_random'] = np.random.choice([True, False], size=len(data),
                                           p=[prop_positive, prop_negative])



def get_evaluation_scores(system, gold):
    """Computing precision and accuracy scores
        given the golden labels column and the system output column"""
    tp = sum(data[system] & data[gold])
    fp = sum(data[system] & ~data[gold])
    tn = sum(~data[system] & ~data[gold])
    fn = sum(~data[system] & data[gold])
    
    t_precision = tp/(tp+fp)
    f_precision = tn/(tn+fn)
    avg_precision = (t_precision + f_precision)/2
        
    accuracy = (tp+tn)/(tp+fp+tn+fn)

    print("{:<50}{:>8.3f}".format('Precision for correctly identifying TRUE sense:', t_precision))
    print("{:<50}{:>8.3f}".format('Precision for correctly identifying FALSE sense:', f_precision))
    print("{:<20}{:>8.3f}".format('Average Precision:', avg_precision))
    print("{:<20}{:>8.3f}".format('Accuracy:', accuracy))
    print()
    


print('Evaluation scores for baseline solutions:')
print()
for prediction in ['baseline_on_frequency', 'baseline_random']:
    print(prediction.upper(), ':')
    get_evaluation_scores(prediction, 'synset_is_correct')


#%%
###############     FEATURE IMPLEMENTATION     ###############



# Feature 1

def get_synset_signature(synset_name):
    """Given a synset name, the function returns, as a joint string,
        the synset definition and all the examples"""
    definition = wordnet.synset(synset_name).definition()
    examples = '. '.join(wordnet.synset(synset_name).examples())
    signature = definition + '. ' + examples
    return signature

data['synset_signature'] = data['synset'].apply(get_synset_signature)


def normalize_sentence(raw_sentence):
    """Given a sentence as a string, the function:
        removes punctuation,
        tokenizes the sentence and removes stop-words,
        returns the stems of the remaining tokens"""
    no_punctuation = re.sub(r"[\.,!?:;\(\)]", " ", raw_sentence)
    no_quotes = re.sub(r"\'{2}|`{2}", " ", no_punctuation)
    tokens = nltk.tokenize.word_tokenize(no_quotes)
    filtered_tokens = [token.lower() for token in tokens if not token.lower() in stop_words]
    stems = [stemmer.stem(word) for word in filtered_tokens]
    return stems


def word_overlap(context, signature):
    """Given two strings, the function normalizes them and
        calculates the amount of words which appear in both of them"""
    norm_context = normalize_sentence(context)
    norm_signature = normalize_sentence(signature)
    overlap = len([token for token in norm_signature if token in norm_context])
    return overlap

data['F1: word_overlap'] = data.apply(
        lambda row: word_overlap(row['full_context'], row['synset_signature']), axis=1)



# Feature 2 - Overlap of bigrams

def bigram_overlap(context, signature):
    """Given two strings, the function return the amount of bigrams of stems
        which appear in both of them (including stop-words and punctuation)"""
    context_tokens = nltk.tokenize.word_tokenize(context.lower())
    signature_tokens = nltk.tokenize.word_tokenize(signature.lower())
    context_stems = [stemmer.stem(token) for token in context_tokens]
    signature_stems = [stemmer.stem(token) for token in signature_tokens]
    context_bigrams = list(bigrams(context_stems))
    signature_bigrams = list(bigrams(signature_stems))
    overlap = len([bigram for bigram in signature_bigrams if bigram in context_bigrams])
    return overlap

data['F2: bigram_overlap'] = data.apply(
        lambda row: bigram_overlap(row['full_context'], row['synset_signature']), axis=1)



# Feature 3 - Similarity of the target word to the average of the synset synonyms

def get_synset_synonyms(synset_name):
    """Given a synset name, returns the list of its synonyms"""
    return wordnet.synset(synset_name).lemma_names()

data['synset_synonyms'] = data['synset'].apply(get_synset_synonyms)


def get_word_to_average_synonyms_vector_similarity(target_word, list_of_synonyms):
    """Given a word and a list of the synonyms,
        the function computes the average vector of the synonyms,
        and returned the cosine of this value and the target word vector"""
    if target_word in words:
        synonyms_vectors = [word2vec[synonym] for synonym in list_of_synonyms
                            if synonym in word2vec.keys()]
        if len(synonyms_vectors)>1:
            synonyms_vectors = np.array(synonyms_vectors)
            synonyms_average_embedding = (np.average(synonyms_vectors, axis=0)).tolist()
            similarity = cosine(word2vec[target_word], synonyms_average_embedding)
        elif len(synonyms_vectors) == 1:
            similarity = cosine(word2vec[target_word], synonyms_vectors[0])
        else:
            similarity = 0
        return similarity

data['F3: COS_word/synonyms'] = data.apply(
    lambda row: get_word_to_average_synonyms_vector_similarity(row['target_word'], row['synset_synonyms']), axis=1)
data.fillna('0', inplace=True)  



# Feature 4 - Similarity between the context and the synset signature

def get_sentence_embedding(sentence):
    """Given a sentence as a string, the function tokenizes it,
        and returns the average of all the vectors (of tokens)"""
    tokens = nltk.tokenize.word_tokenize(sentence)
    embeddings = [word2vec[token] for token in tokens if token in word2vec.keys()]
    embeddings = np.array(embeddings)
    sentence_embedding = (np.average(embeddings, axis=0)).tolist()
    return sentence_embedding


def get_sentences_cosine(context, synset_signature):
    """Given two strings, returns the cosine of the embeddings of these strings"""
    context_embedding = get_sentence_embedding(context)
    signature_embedding = get_sentence_embedding(synset_signature)
    similarity = cosine(context_embedding, signature_embedding)
    return similarity

data['F4: COS_context/signature'] = data.apply(
    lambda row: get_sentences_cosine(row['full_context'], row['synset_signature']), axis=1)


#%%    
###############     LOGISTICS REGRESSION     ###############



data_train, data_test = train_test_split(data, test_size=0.2, random_state=0)

X_train = data_train[['F1: word_overlap', 'F2: bigram_overlap',
                      'F3: COS_word/synonyms', 'F4: COS_context/signature']]
X_test = data_test[['F1: word_overlap', 'F2: bigram_overlap',
                    'F3: COS_word/synonyms', 'F4: COS_context/signature']]

y_train = data_train['synset_is_correct']
y_test = data_test['synset_is_correct']


def logistic_regression(X_train, X_test, y_train, y_test):
    
    #model = LogisticRegression(class_weight='balanced')
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
        
    print("LOGISTIC REGRESSION SUCCESSFULLY IMPLEMENTED!")
    print()

    conf_matrix = confusion_matrix(y_test, y_pred)
    tn = conf_matrix[0, 0]
    tp = conf_matrix[1, 1]
    fn = conf_matrix[1, 0]
    fp = conf_matrix[0, 1]
    t_precision = tp/(tp+fp)
    f_precision = tn/(tn+fn)
    avg_precision = (t_precision + f_precision)/2

    print("{:<50}{:>8.3f}".format('Precision for correctly identifying TRUE sense:', t_precision))
    print("{:<50}{:>8.3f}".format('Precision for correctly identifying FALSE sense:', f_precision))
    print("{:<15}{:>8.3f}".format('AVG Precision:', avg_precision))
    print("{:<15}{:>8.3f}".format('Accuracy:', model.score(X_test, y_test)))
    print()
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print()
    print("Model coefficients:")
    for idx, col_name in enumerate(X_train.columns):
        print("{:<25} --> {:>8.3f}".format(col_name, model.coef_[0][idx]))
    print()
    
    return y_pred

y_pred = logistic_regression(X_train, X_test, y_train, y_test)







