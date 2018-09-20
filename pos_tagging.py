import numpy as np
import pandas as pd
import sqlite3
from sklearn.feature_extraction.text import CountVectorizer

conn = sqlite3.connect("full_table.db")
c = conn.cursor()
df = pd.read_sql_query("select * from all_words;", conn)

conn.close()

def remove_brackets(entry):
    if entry != None:
        new = entry.strip("[]")
        if (entry != new) and (new != "'word not found'"):
            new = new.split(', ', 1)
            return new
        else:
            return [new]
    else:
        return ["word not found"]

new_pos = df["pos"].apply(remove_brackets)

new_pos = pd.DataFrame(new_pos.values.tolist(), columns=['pos1','pos2'])

def remove_quotes(entry):
    if entry != None:
        new = entry.strip("''")
        return new
    else:
        return "no second POS"

full_table = pd.DataFrame(dict(pos1 = new_pos["pos1"].apply(remove_quotes), pos2 = new_pos["pos2"].apply(remove_quotes), words = df["words"])).reset_index()

full_table = full_table.drop_duplicates(subset = "words", keep = "first")

not_found_words = full_table.loc[full_table["pos1"] == "word not found"]

full_table_wo_none = full_table.loc[full_table["pos1"] != "word not found"].dropna(axis = 0)

print(full_table.head)


def split_the_word_for_morphems(word):
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
            break
        except:
            prefix1 = word[0:2]
            prefix2 = word[0:3]
            suffix1 = word[-1:]
            suffix2 = word[-2:]
            suffix3 = word[-3:]
            return " ".join([prefix1, prefix2, suffix1, suffix2, suffix3])

x_train = full_table_wo_none.loc[0:60000, "words"].apply(split_the_word_for_morphems)
y_train = full_table_wo_none.loc[0:60000, "pos1"]
x_test = full_table_wo_none.loc[60000:, "words"].apply(split_the_word_for_morphems)
y_test = full_table_wo_none.loc[60000:, "pos1"]

'''
bag_of_words = list(x_train.apply(pd.Series).stack().value_counts().index)

def vectorizing(word_with_morphemes):
    word_vector = np.zeros(len(bag_of_words))
    for i in range(len(bag_of_words)):
        if bag_of_words[i] in word_with_morphemes:
            word_vector[i] = 1
        else:
            word_vector[i] = 0
    return word_vector

x_train_vec = x_train.apply(vectorizing)
print(x_train_vec)
'''

vectorizer = CountVectorizer()
#vectorizer.fit(x_train)
x_train_vec = vectorizer.fit_transform(x_train.tolist())
x_test_vec = vectorizer.transform(x_test.tolist())
print(x_train_vec.shape)
print(y_train.shape)
print(x_test_vec.shape)
print(y_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

models = [LinearSVC(), LogisticRegression(), RandomForestClassifier()]
def est_error(model):
    model.fit(x_train_vec, y_train)
    y_pred = model.predict(x_test_vec)
    comparison = pd.DataFrame(dict(words = full_table_wo_none.loc[60000:, "words"], predicted = y_pred, real = y_test))
    error = sum(comparison.apply(lambda x: 1 if x["predicted"] != x["real"] else 0, axis = 1))/len(y_pred)
    wrong_predictions = comparison.apply(lambda x: str(x["words"]) + str(x["predicted"]) + str(x["real"]) if x["predicted"] != x["real"] else None, axis = 1)
    return error, wrong_predictions

for model in models:
    error, predictions = est_error(model)
    print(predictions.dropna(axis = 0))



