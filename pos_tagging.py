import pandas as pd
import numpy as np
import sqlite3
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from datetime import datetime

# Opening connection to the sql database with all scrapped words + POS and loading as a DataFrame
conn = sqlite3.connect("full_table.db")
c = conn.cursor()
df = pd.read_sql_query("select * from cleaned_all_words;", conn)
conn.close()

print(df)


# Preparing the words for part-of-speech machine learning tags
def split_the_word_for_morphems(word):
    '''splitting words in possible morphemes'''
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


# Splitting the dataset in training and test data
x_train_words, x_test_words, y_train, y_test = train_test_split(df.loc[:, "words"], df.loc[:, "pos1"], test_size=0.3, random_state=0)

# Splitting words in parts/ possible morphems
x_train = x_train_words.apply(split_the_word_for_morphems)
x_test = x_test_words.apply(split_the_word_for_morphems)

# Preparing data for machine learning tasks, vectorizing words, transforming sparse matrix to ordinary one
vectorizer = CountVectorizer()
x_train_vec = vectorizer.fit_transform(x_train.tolist()).toarray()
x_test_vec = vectorizer.transform(x_test.tolist()).toarray()

# Creating manually dimensions to the vectors: length of the word and whether the first letter is a capital letter
len_words_train = x_train_words.apply(lambda x: len(x))
len_words_test = x_test_words.apply(lambda x: len(x))

isupper_train = x_train_words.apply(lambda x: 1 if x[0].isupper() is True else 0)
isupper_test = x_test_words.apply(lambda x: 1 if x[0].isupper() is True else 0)

shapes = (len_words_train.shape[0], len_words_test.shape[0])

# Combining manually created features with automatically vectorized features
x_train_vec = csr_matrix(np.hstack([x_train_vec, len_words_train.values.reshape(shapes[0], 1), isupper_train.values.reshape(shapes[0], 1)]))
x_test_vec = csr_matrix(np.hstack([x_test_vec, len_words_test.values.reshape(shapes[1], 1), isupper_test.values.reshape(shapes[1], 1)]))

# Machine learning part. Trying several techniques
models = [LogisticRegression(), RandomForestClassifier(), LinearSVC(C=0.01), LinearSVC(C = 10), LinearSVC(C = 0.1)]


def est_accuracy(model):
    '''calculating estimated accuracy of a model'''
    model_train = model.fit(x_train_vec, y_train)
    accuracy = model_train.score(x_test_vec, y_test)
    return "%0.2f" % accuracy


# Printing the accuracies of different models
for model in models:
    startTime = datetime.now()
    print("Accuracy of " + str(model.__class__.__name__) + " = " + str(est_accuracy(model)))
    print("Time to implement method: " + str(datetime.now()-startTime))
