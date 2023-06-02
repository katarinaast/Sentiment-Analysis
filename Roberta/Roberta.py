from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import pandas as pd

#ucitavamo model i tokenizer
roberta = "cardiffnlp/twitter-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)

labels = ['Negative', 'Neutral', 'Positive']
tweets = []

#ucitavamo podatke
data = pd.read_csv("C:\\Users\\katarina.stanojkovic\\source\\repos\\Roberta\\test.csv", delimiter= ',')

#uzimamo samo vrednosti tvitova, odnosno tekst
for row in data.iterrows():
    r=row[1]['tweet']
    tweets.append(r)


for tweet in tweets:
    tweet_words = []
    #razdvajamo reci teksta 
    for word in tweet.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
    
        elif word.startswith('http'):
            word = "http"
        tweet_words.append(word)
    #ovako promenjene reci se spajaju u tweet_proc
    tweet_proc = " ".join(tweet_words)
    #kreira se tenzor, sadrzi input - tvit preveden u brojni niz, attention mask - koje reci su bitnije
    encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
    print(tweet_proc)
    #modelu prosledjujemo kljuceve i vrednosti encoded_tweet 
    output = model(**encoded_tweet)
    #ono sto dobijemo pretvaramo u numpy niz
    scores = output[0][0].detach().numpy()
    #racunamo verovatnocu za svaku labelu
    scores = softmax(scores)

    for i in range(len(scores)):
    
        l = labels[i]
        s = scores[i]
        print(l,s)





