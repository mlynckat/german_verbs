import pandas as pd
import numpy as np
import sqlite3
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from datetime import datetime


# Opening connection to the sql database with all cleaned words + POS and loading as a DataFrame
conn = sqlite3.connect("full_table.db")
c = conn.cursor()
df = pd.read_sql_query("select * from cleaned_all_words;", conn)
conn.close()


# Preparing the words for part-of-speech machine learning tags
def split_the_word_for_morphems(word):
    """splitting words in possible morphemes"""
    while True:
        try:
            prefix1 = word[0:2]
            prefix2 = word[0:3]
            prefix3 = word[2:4]
            prefix4 = word[3:5]
            suffix1 = word[-1:]
            suffix2 = word[-2:]
            suffix3 = word[-3:]
            suffix4 = word[-4:]
            return " ".join([prefix1, prefix2, prefix3, prefix4, suffix1, suffix2, suffix3, suffix4])
        except Exception as e:
            print(e)
            prefix1 = word[0:2]
            prefix2 = word[0:3]
            suffix1 = word[-1:]
            suffix2 = word[-2:]
            suffix3 = word[-3:]
            return " ".join([prefix1, prefix2, suffix1, suffix2, suffix3])


all_x_words = df.loc[:, "words"]
all_x = all_x_words.apply(split_the_word_for_morphems)

# Preparing data for machine learning tasks, vectorizing words, transforming sparse matrix to ordinary one
vectorizer = CountVectorizer()
all_x_vec = vectorizer.fit_transform(all_x.tolist()).toarray()

# Creating manually dimensions to the vectors: length of the word and whether the first letter is a capital letter
len_all_x = all_x_words.apply(lambda x: len(x))
isupper_all_x = all_x_words.apply(lambda x: 1 if x[0].isupper() is True else 0)

# Combining manually created features with automatically vectorized features
all_x_vec = csr_matrix(np.hstack([all_x_vec, len_all_x.values.reshape(73048, 1), isupper_all_x.values.reshape(73048, 1)]))

# Machine learning part. Definition of the models to use
models = [LinearSVC(), LinearSVC(C=10), LinearSVC(C=0.1), LinearSVC(C=0.01), LogisticRegression(), RandomForestClassifier()]


def est_accuracy(model):
    """calculating estimated accuracy of a model"""
    accuracy = cross_val_score(model, all_x_vec, df.loc[:, "pos1"], cv=5)
    return "%0.2f (+/- %0.2f)" % (accuracy.mean(), accuracy.std() * 2)


for model in models:
    startTime = datetime.now()
    print("Accuracy of " + str(model.__class__.__name__) + " = " + str(est_accuracy(model)))
    print("Time to implement method: " + str(datetime.now()-startTime))
