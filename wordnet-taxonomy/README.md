`wordnet_main.py` consists of 2 parts: **Hyponymy Chains** and **WordNet Taxonomic Relations with word2vec**

## Hyponymy Chains

1. Pass any word to `print(wordnet.synsets("study"))` to know its synsets (here, *'study'* as an example);
2. Pass any synset to `synset = wordnet.synset('study.n.05')` to see all hypernym chains for it (here, *'study.n.05'* as an example);
3. Select any hypernym chain. The lemmas of all hypernyms will be printed bottom-up.

Example for ***'study.n.05'***: 
study ( a room used for reading and writing and studying )
 => room
   => area
     => structure, construction
       => artifact, artefact
         => whole, unit
           => object, physical_object
             => physical_entity
               => entity
               

## WordNet Taxonomic Relations with word2vec

**Embedded representations of words** (word2vec) and **cosine as a similarity metric** are used to test (and confirm) the following hypotheses:
1. (a) On average, a word is more similar to its synonyms than to its hypernyms;
   (b) On average, a word is more similar to its hypernyms than to its hyponyms.
2. On average, a word is more similar to the definition of its most frequent sense (i.e., first synset), than to definitions of its less frequent senses (implemented with sentence embeddings).
