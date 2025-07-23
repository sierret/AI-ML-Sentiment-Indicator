# AI-ML-Sentiment-Indicator

N.B. Pyspark version is a WIP due to lacking TFIDF features.

Analyses text to determine if sentiment is positive or negative(or neutral). The text is currently obtained from the top results of a Google Search based on an input string to be used as a search query.
Utilizes Vader lexicon to determine text polarity score that is then used to produce a label. Text is then placed in vectors using TFIDF to function as features. Data split, and model fit and tested on split pars as per usual. Currently uses Bayes algo(Multimodal).

Produces:
Classification Report(precision,recall,f1-score,support)
Confusion Matrix

Note: There are two versions, one based around scikit-learn and the other pyspark. Pyspark version uses scikit-learn[for now] for metrics only; no core model-training usage AND is currently a WIP. scikit-learn version is functional and independent of pyspark. This is because Pyshark does not offer TFIDF vectorization out-of-the-box. Thus, I am currently implementing the functionality myself, intending to use the HashTF and IDF algorithms separately and sequentially to achieve the TFIDF.

Requires:
scikit-learn
pyspark[version-dependent]
nltk
googlesearch-python
pandas

