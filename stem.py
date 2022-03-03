from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
import re

class stem():
    def stem_tokenizer(self, text):
        stemmer = EnglishStemmer(ignore_stopwords=True)
        words = re.sub(r"[^A-Za-z0-9\-]", " ", text).lower().split()
        words = [stemmer.stem(word) for word in words]
        return words
