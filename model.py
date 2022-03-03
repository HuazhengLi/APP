import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
from st import st
import re

tfidf = TfidfVectorizer(stop_words=stopwords.words('english'),
                        tokenizer=st().stem_tokenizer,
                        lowercase=True,
                        max_df=0.9,
                        min_df=5,
                        ngram_range=(1, 5)
                       )

df = pd.read_csv("news_class.csv")
X = df.headline
y = df.pc_class  # 1: stock price changed >=|0.03|, 0: stock price change <|0.03|
model = Pipeline([('tfidf', tfidf), ('lg', LogisticRegression(class_weight='balanced'))])
model.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
