"""
WordNet and Word Embeddings
"""


import pandas as pd
import numpy as np
import nltk
import re
import random
from nltk.corpus import wordnet


#%%
###############     HYPONYMY CHAINS     ###############


print(wordnet.synsets("study")) # check any word for available synsets
print()
synset = wordnet.synset('study.n.05') # pass any word for exploration
print(synset.hypernym_paths()) # see all hypernym chains available for the synset
print()

chain = synset.hypernym_paths()[0]  # select any chain


# Pretty print the hierarcy
space = '  '
arrow = '=>'
n = 0

print(', '.join(map(str, synset.lemma_names())), '(',synset.definition(),')')
for item in chain[-2::-1]:
    print(space*n, arrow, ', '.join(map(str, item.lemma_names())))
    n += 1
print()


#%%
###############     WORD2VEC DATA SETUP     ###############



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



# Option to use full dataset or its subset
full_data = True
if full_data:
    words_sample = words
    vectors_sample = vectors
    print('FULL DATASET FROM WORD2VEC IS USED!')
else:
    words_sample = words[:5000]
    vectors_sample = vectors[:5000]
    print('A SUBSET OF WORD2VEC DATA IS USED!')
print()



# Putting words and vectors to a dataframe for easier usage
# One dataframe per task (hypothesis)
data1 = pd.DataFrame(list(zip(words_sample, vectors_sample)), columns = ['Word', 'Vector'])
data2 = data1.copy()


#%%
###############     WORD EMBEDDINGS     ###############



# HYPOTHESIS 1:
# (a) on average, a word is more similar to its synonyms than to its hypernyms;
# (b) on average, a word is more similar to its hypernyms than to its hyponyms.


def get_all_synonyms(word):
    """Given a word, the function returns the list of
        all its one word synonyms, from all the synsets
        (multiple word examples are not present in word2vec data)"""
    word_senses = wordnet.synsets(word)
    one_word_synonyms = []
    for synset in word_senses:
        for lemma in synset.lemma_names():
            if (lemma != word) and (lemma not in one_word_synonyms) and ('_' not in lemma):
                one_word_synonyms.append(lemma)
    return one_word_synonyms


def get_all_hypernyms(word):
    """Given a word, the function returns the list of
        all its one word hypernyms, from all the synsets"""
    word_senses = wordnet.synsets(word)
    string_of_hypernyms = ""
    for synset in word_senses:
        for hypernym in synset.hypernyms():
            string_of_hypernyms = string_of_hypernyms + " " + str(hypernym)
    all_hypernyms = re.findall(r'(\w+)(?=\.[nvasr]{1}\.\d{2})', string_of_hypernyms)
    one_word_hypernyms = []
    for hypernym in all_hypernyms:
        if (hypernym != word) and ('_' not in hypernym) and (hypernym not in one_word_hypernyms):
            one_word_hypernyms.append(hypernym)
    return one_word_hypernyms


def get_all_hyponyms(word):
    """Given a word, the function returns the list of
        all its one word hyponyms, from all the synsets"""
    word_senses = wordnet.synsets(word)
    string_of_hyponyms = ""
    for synset in word_senses:
        for hyponym in synset.hyponyms():
            string_of_hyponyms = string_of_hyponyms + " " + str(hyponym)
    all_hyponyms = re.findall(r'(\w+)(?=\.[nvasr]{1}\.\d{2})', string_of_hyponyms)
    one_word_hyponyms = []
    for hyponym in all_hyponyms:
        if (hyponym != word) and ('_' not in hyponym) and (hyponym not in one_word_hyponyms):
            one_word_hyponyms.append(hyponym)
    return one_word_hyponyms


def get_similarities(target_vector, related_words):
    """Given a vector of the target word and a list of other words
        (synonyms, hypernyms or hyponyms), the function returns a list of pairs:
        a word from a list and its similarity to the target word"""
    word_cosine_pairs = []
    if related_words:   # verifying it's not an empty list
        for word in related_words:
            if word in word2vec.keys():
                word_vector = word2vec[word] 
                similarity_to_target = cosine(target_vector, word_vector)
                word_cosine_pairs.append((word, similarity_to_target))
    return word_cosine_pairs
                

def test_hypothesis_1(df):
    """The function tests 2 hypotheses:
        (a) on average, a word is more similar to its synonyms than to its hypernyms;
        (b) on average, a word is more similar to its hypernyms than to its hyponyms"""
    
    df['Synonyms'] = df['Word'].apply(get_all_synonyms)
    df['Hypernyms'] = df['Word'].apply(get_all_hypernyms)
    df['Hyponyms'] = df['Word'].apply(get_all_hyponyms)
    
    df['SYN Cosines'] = df.apply(
        lambda row: get_similarities(row['Vector'], row['Synonyms']), axis=1)
    df['SYN avg_cos'] = df['SYN Cosines'].apply(
        lambda x: np.mean([cos for (word, cos) in x]))

    df['HYPER Cosines'] = df.apply(
        lambda row: get_similarities(row['Vector'], row['Hypernyms']), axis=1)
    df['HYPER avg_cos'] = df['HYPER Cosines'].apply(
        lambda x: np.mean([cos for (word, cos) in x]))

    df['HYPON Cosines'] = df.apply(
        lambda row: get_similarities(row['Vector'], row['Hyponyms']), axis=1)
    df['HYPON avg_cos'] = df['HYPON Cosines'].apply(
        lambda x: np.mean([cos for (word, cos) in x]))

    print('The average values of cosines:')
    print(df[['SYN avg_cos', 'HYPER avg_cos', 'HYPON avg_cos']].mean())
    print()

test_hypothesis_1(data1)


#%%
###############     SENTENCE EMBEDDINGS     ###############



# HYPOTHESIS 2:
# on average, a word is more similar to the definition of its most frequent sense
# (i.e., first synset), than to definitions of its less frequent senses


def get_sentence_embedding(sentence):
    """Given a sentence as a string, the function:
        tokenizes it, looks for a vector for each token,
        and returns the sentence embedding - the average of all vectors
        (note: not all of them are present in word2vec)"""
    tokens = nltk.tokenize.word_tokenize(sentence)
    embeddings = [word2vec[token] for token in tokens if token in word2vec.keys()]
    embeddings = np.array(embeddings)
    sentence_embedding = (np.average(embeddings, axis=0)).tolist()
    return sentence_embedding


def get_synset_definition(synset):
    """Given a synset name, the function returns its definition"""
    return synset.definition()


def test_hypothesis_2(df):
    """The function tests the following hypothesis:
        on average, a word is more similar to the definition
        of its most frequent sense (i.e., first synset), than to definitions
        of its less frequent senses"""
    
    df['First synset definition'] = df['Word'].apply(
        lambda x: get_synset_definition(wordnet.synsets(x)[0]))
    df['First def vector'] = df['First synset definition'].apply(get_sentence_embedding)
    
    df['Random synset definition'] = df['Word'].apply(
        lambda x: get_synset_definition(random.choice(wordnet.synsets(x)[1:])))
    df['Random def vector'] = df['Random synset definition'].apply(get_sentence_embedding)
    
    df.dropna(inplace=True)     # Some of the definitions will not have vectors
                                # because none of its tokens is in word2vec.
                                # We remove these rows to avoid further errors
                                # (we cannot compute cosines if there is no vector)
    
    df['COS to first def'] = df.apply(
        lambda row: cosine(row['Vector'], row['First def vector']), axis=1)
    df['COS to random def'] = df.apply(
        lambda row: cosine(row['Vector'], row['Random def vector']), axis=1)
    
    print('The average similarity of a word to the definitions of its first and random senses:')
    print(df[['COS to first def', 'COS to random def']].mean())
    print()


# To test hypothesis 2, a word needs to have at least 2 synsets
# So, we remove the words which have 1 or no synsets at all 
data2['Word has at least 2 synsets'] = data2['Word'].apply(
    lambda x: True if len(wordnet.synsets(x))>1
    else False)
data2.drop(data2.index[data2['Word has at least 2 synsets'] == False], inplace=True)

test_hypothesis_2(data2)


