# AI-ML-Sentiment-Indicator
Analyses text to determine if sentiment is positive or negative(or neutral)
Utilizes Vader lexicon to determine text polarity score that is then used to produce a label. Text is then placed in vectors using TFIDF to function as features. Data split, and model fit and tested on split pars as per usual. Currently uses Bayes algo(Multimodal).

Produces:
Classification Report(precision,recall,f1-score,support)
Confusion Matrix

Note: There are two versions, one based around scikit-learn and the other pyspark. Pyspark version uses scikit-learn[for now] for metrics only; no core model-training usage. scikit-learn version is independent of pyspark.

Requires:
scikit-learn
pyspark[version-dependent]
nltk
googlesearch-python
pandas

NOTE: pyspark does not offer out-of-the-box TFIDF vectorizer, unlike pyspark. As such, it used HashTF and IDF algorithms separately and sequentially.
