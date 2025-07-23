
from googlesearch import search
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk

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
    
    print(len(titles))
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(titles)
    #y = [1]  # sentiment label

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Evaluate model performance
    y_pred = model.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Classification Report:')
    print(classification_report(y_test, y_pred))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))

