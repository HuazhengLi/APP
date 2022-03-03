from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
import re
class st():	
    def stem_tokenizer(self, text):
        stemmer = EnglishStemmer(ignore_stopwords=True)
        words = re.sub(r"[^A-Za-z0-9\-]", " ", text).lower().split()
        words = [stemmer.stem(word) for word in words]
        return words

#if __name__ == "__main__":
#	f = st()
#	print (stem_tokenizer(text))