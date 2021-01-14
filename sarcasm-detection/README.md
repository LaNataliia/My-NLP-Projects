# Short Project Description

The main code file is `sarcasm_main.py`, the working dataset is `sarcasmdata.xlsx`.

### The workflow

1. Loading `sarcasmdata.xlsx` into a dataframe: columns **tweet** and **class** (regular/ sarcastic), 39247 tweets in total (20678 sarcastic, 18569 regular)
2. Data cleaning and basic statistics (incl. Countplot, WordCloud)
3. Training a Logistic Regression model for classifying tweets into sarcastic and not sarcastic in 3 steps:

  3.1 Implementation of 7 basic features: number of tokens, number of hashtags (excl. *#sarcasm*), number of exclamation/ punctuation marks as numeric features, and user mentions,     interjections, tripled letters, capitalized words as Boolean features
  3.2 Adding sentiment as a feature using **TextBlob** (TextBlob sentiment, and TextBlob subjuctivity)
  3.3 Adding 7 advanced features based on sentiment: number of positive words, number of negative words, standard deviation of positive and negative scores of all words per tweet     (scores obtained from **SentiWordNet**), difference between the maximum and the average scores (based on pos/neg scores of all words per tweet) as numeric features, and           polarity change in the beginning and the end of a tweet based on **SentiWordNet** scores, polarity change based on positive/negative words count, negative average score -         positive maximum score (based on pos/neg scores of all words per tweet) as Boolean features
  
### The results

1. The model performance reaches **82.1% on accuracy** (precision 82.09%, recall 86.18%, f1-score 84.09%)
2. Sentiment carries relevant information which can facilitate sarcasm detection model; however, it should be involved not as a single feature, but as a combination of multiple sentiment-related features

The full project description is available: `research-project-full-description.pdf`
