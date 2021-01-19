"""
Generate a Song
"""

import sys
import json
import nltk
import random

from nltk import FreqDist
from nltk import word_tokenize
from nltk.corpus import *
from nltk.util import bigrams, trigrams, ngrams



def save_ngrams(lyrics_file):
    """Creates a file of N-grams and their occurences for the file with lyrics"""
    all_ngrams = []
    
    with open(lyrics_file, "r") as f:
        for line in f.readlines():
            if not line.isupper() and line != "\n":
                all_ngrams.extend(list(nltk.bigrams(word_tokenize(line), pad_left=True, pad_right=True, right_pad_symbol='</s>', left_pad_symbol='<s>')))
    
    fd = FreqDist(all_ngrams)
    ngram_stats = fd.most_common(fd.N())
    
    with open('ngrams.json', mode='w') as fp:
        json.dump(ngram_stats, fp)
        
        

def generate_word(ngrams, word):
    """Given a word, generate one next word"""
    
    possible_ngrams = []
    for ngram in ngrams:

        if ngram[0][0] == word:
            possible_ngrams.append(ngram)
    
    if len(possible_ngrams) > 5:
        next_ngram = random.sample(possible_ngrams[:5], 1)
    elif len(possible_ngrams) == 0 and word != "</s>":
        next_word = "</s>"
        # print (possible_ngrams)
        return next_word
    elif len(possible_ngrams) == 0 and word == "</s>":
        next_word = "<s>"
        # print (possible_ngrams)
        return next_word
    else:
        next_ngram = random.sample(possible_ngrams, 1)
    
    # print (next_ngram)
    
    next_word = next_ngram[0][0][1]
    return next_word



def generate_song(word):
    
    with open('ngrams.json', mode='r') as fp:
        ngrams = json.load(fp)
    
    song = [word]
    lines = 0
    
    for i in range(5000):
        nextword = generate_word(ngrams ,word)
        song.append(nextword)
        prev_word = word
        word = nextword
        
        if nextword == "</s>":
            lines += 1
            
        # if lines == 5:
        #     song.append("\n")
            
        if lines > 21:
            break
        
    song = " ".join(song).replace("</s> <s> ", "\n")
    song = song[:-5]
    
    return song

  

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: " + sys.argv[0] + " <name of the lyrics file> <first word of the song>")
    else:
        save_ngrams(sys.argv[1])
        print(generate_song(sys.argv[2]))
        