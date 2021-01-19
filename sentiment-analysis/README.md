Sentiment analysis is performed on the dataset of the movie reviews, `sentiment.csv`. The dataset contains **20.000 reviews**, the classes are ballanced: 10.000 positive and 10.000 negative. The Logistic Regression model for this task of sentiment classification is trained in **2 different ways**: with the hand-crafted features, and with the bag-of-words representation.


## Logistic Regression with the Hand-Crafted Features

The working file: `sentiment_features.py`

The features are:
1. amount of positive words,
2. amount of negative words (on the basis of the Opinion Lexicon by Liu&Hu, 2004: `positive-words.txt`, `negative-words.txt`),
3. ratio of adjectives (to all tokens, excluding stopwords),
4. average sentence length in tokens,
5. the filtered text ratio (ratio of the tokens excluding stopwords to all the tokens).

The model performs with **72.6% accuracy**.

An additional feature of the SentiWordNet (positive and negative scores of each word added up and substracted respectively to result in one single score per review) **increases the accuracy till 73.0%**.


## Logistic Regression with the Bag-of-Words Representation

The working file: `sentiment_bow.py`

Bag-of-Words is implemented in 3 different ways: with CountVectorizer for unigrams and bigrams, and with Tfidf Vectorizer.

The results are different for raw and for normalized text.
1. For normalized text:
   - CountVectorizer, unigrams: **86.8% accuracy**
   - CountVectorizer, bigrams: **84.3% accuracy**
   - TfidfVectorizer: **88.4% accuracy**
2. For raw text:
   - CountVectorizer, unigrams: **87.0% accuracy**
   - CountVectorizer, bigrams: **87.0 accuracy**
   - TfidfVectorizer: **88.2% accuracy**

Text normalization is an important step as it removes 'noise' from the text and prevents the model from overfitting. However, when working with bigrams and tf-idfs, it is better to look at raw text because by filtering out stopwords, we may also lose important information (e.g. prepositions can change the meaning to the opposite in the phrasal verbs).
