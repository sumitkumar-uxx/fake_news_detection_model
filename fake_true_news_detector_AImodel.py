import pandas as pd #importing pandas
import numpy as np # imporsting numpy
from sklearn.linear_model import LogisticRegression #importing logistiR from liner.model
from sklearn.feature_extraction.text import TfidfVectorizer #use it for taking the input as the text

 
am = pd.read_csv(r"C:\Users\sumit\OneDrive\Documents\fake.csv")  # fake news
a = pd.read_csv(r"C:\Users\sumit\OneDrive\Documents\true.csv")   # true news

am['label'] = 0
a['label'] = 1

df = pd.concat([am, a])
df = df.sample(frac=1).reset_index(drop=True)  
df['content'] = df['title'] + " " + df['text']

inputs = df['content']
output = df['label']

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X = vectorizer.fit_transform(inputs)


op = LogisticRegression()
op.fit(X,output)
title_input = input("Enter your news title: ")
text_input = input("Enter your news text: ")
userinput = title_input + " " + text_input
vector = vectorizer.transform([userinput])
result = op.predict(vector)

if result[0] == 0:
    print("This news is likely FAKE 📰❌")
else:
    print("This news is likely TRUE 🗞✅")
 