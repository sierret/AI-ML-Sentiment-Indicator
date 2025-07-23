
from googlesearch import search
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from pyspark.sql import SparkSession
import pandas as pd
from pyspark.ml.feature import Tokenizer,HashingTF,IDF
text= "trump"

if __name__=="__main__":
    nltk.download('vader_lexicon')
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    vds=SentimentIntensityAnalyzer()
    titles=[]
    y=[]
    for result in search(text, advanced=True,num_results=4000): #Making ```advanced=True``` shows the title, description, and URL of all results.
        title=(result.title)
        #print(title)
        title = title.lower()
        title = title.replace('\W', ' ')
        #print((result))
        #print((title))
        #print(vds.polarity_scores(title))
        if vds.polarity_scores(title)['compound']==0:
            continue
        elif vds.polarity_scores(title)['compound']<0:
            y.append(0)
        else:
            y.append(1)
        titles.append(title)
        if (len(titles)==100):
            break
    
    df = pd.DataFrame(titles, columns=['titles'])
    spark = SparkSession.builder.appName('example').getOrCreate()
    data=spark.createDataFrame(df)
    tokenizer = Tokenizer(inputCol="titles", outputCol="words")
    words=tokenizer.transform(data)
    hashingTF = HashingTF(inputCol="words", outputCol="wordFeatures")
    hashTFData = hashingTF.transform(words)

    idf=IDF(inputCol="wordFeatures", outputCol="features")
    idfModel = idf.fit(hashTFData)
    tdIdfVecs = idfModel.transform(hashTFData)
    
    #y = [1]  # sentiment label

    
    new_data = [row + (l,) for row, l in zip(tdIdfVecs, y)]

    # Create a new DataFrame with the additional column
    new_columns = tdIdfVecs.columns + ["label"]
    tdIdfVecs.select("features").show()
    data = spark.createDataFrame(new_data, new_columns)
    # Split data into training and testing sets
    train_data, test_data = data.randomSplit([0.7, 0.3], seed=42)
    # Train Naive Bayes model
    nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
    nb = nb.fit(train_data)

    # Evaluate model performance
    y_pred = nb.transform(test_data)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Classification Report:')
    print(classification_report(y_test, y_pred))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))

