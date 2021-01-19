Word sense disambiguation task is performed on SemCor dataset `semcor.csv`, and consists in classifying a given synset as correct or incorrect for the target word in the given context. The dataset contains 50.001 entries with unballanced classes: only 1/3 of the synsets are correct, and 2/3 are incorrect (for each target word-context pair there are 3 options of synsets, and only 1 is correct).


### Baseline Solutions

- **Baseline 1**: always select the most frequent synset of the target word, i.e. the first synset listed in WordNet. The macroaverage precision of this solution is **79.3%**.
- **Baseline 2**: assign random True/False value for a given synset taking into account the proportion of gold labels (1/3 vs 2/3). The macroaverage precision of this solution is **around 50%**.


### Logistic Regression with Word and Sentence Embeddings

The model for classifying synsets is trained with the following features:
1. Word overlap: amount of words which overlap in the given context and in the signature of the chosen synset. The synset signature includes the definition and the examples. This count is done on stems and excludes punctuation and stop-words.
2. Bigram overlap: Similar to feature 1, but it takes into account the bigrams of stems (stop-words and punctuation included).
3. Similarity between the target word and the synonyms of the chosen synset: cosine of the target word vector and the average vector of its synonyms.
4. Similarity between the context and the signature of the chosen synset: cosine of the context embedding and the chosen synset signature (definition + examples) embedding.

**The macroaverage precision of this solution is 62.1% (accuracy: 67.1%).**


### Interpretation of the results
- The most relevant feature - the similarity between the context and the synset signature. In general, a sentence embedding is not a very precise representation of the sentence meaning since it does not take into account the word order and the ways the words are combined into bigger units. However, the coefficient of this feature suggests that it is anyway more informative to look at the “environments” (even if it is just “a bag of (independent) vectors”) and their embeddings rather than at separate words in order to clarify the meaning in which the word is used.
- Looking at separate words shows different results in two features (feature 1 and feature 3). Comparing embeddings of separate words is less effective than counting overlaps of separate words. This highlights the importance of the context: some words tend to co-appear often and this tendency will be brought up when counting overlaps across different contexts. However, vector similarity between synonyms might not be accurate in this particular model because not all synonyms have vectors in word2vec and some lists of synonyms only include the same lemma as the target word.
- Bigram overlap has a lower coefficient than word overlap, although usually bigrams carry more meaning and information than separate words (they can be viewed as mini-contexts). But in this case we don’t work with long texts, so perhaps this is not enough for getting significant results on bigram statistics.
