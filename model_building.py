import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from stem import stem
import pickle
from nltk.corpus import stopwords

df = pd.read_csv('news_pc_app.csv')
X = df.headline
y = df.pc_class

tfidf = TfidfVectorizer(stop_words = stopwords.words('english'),
                        tokenizer =stem().stem_tokenizer,
                        lowercase = True,
                        max_df = 0.9,
                        min_df = 5,
                        ngram_range = (1, 3))
model = Pipeline([('tfidf', tfidf), ('lg', LogisticRegression(random_state = 2022, class_weight = 'balanced', max_iter = 1000))])
model.fit(X, y)
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f) 
