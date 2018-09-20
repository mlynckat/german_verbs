import duden
import pandas as pd
import numpy as np
import re

#reading the words in a list and converting in a Pandas Series
f = open("list_of_words.txt","r")
list_of_words_unparsed = []
for line in f:
    list_of_words_unparsed.append(line)
list_of_words_unparsed = list_of_words_unparsed[45:len(list_of_words_unparsed)]

series_of_words_unparsed = pd.Series(list_of_words_unparsed)

#Parsing the words
def extract_words(series):
    ''' function that parses the words/ cleans'''
    searched = re.search("\w+", series)
    new_entry = searched.group(0)
    return new_entry

series_of_words = series_of_words_unparsed.apply(extract_words)

#Selecting verbs with prefix aus-
aus_verbs_bool = np.zeros(len(series_of_words), dtype = bool)
for i in range(0, len(series_of_words)):
    aus_search = re.match("^aus.*en$", series_of_words[i])
    if aus_search:
        aus_verbs_bool[i] = True

aus_verbs_series = pd.Series(np.asarray(series_of_words[aus_verbs_bool]), index = range(0, len(series_of_words[aus_verbs_bool])))

#replacing umlauts
aus_verbs_series = aus_verbs_series.apply(lambda x: x.replace("ö", "oe").replace("ä", "ae").replace("ü", "ue").replace("ß", "ss"))

#getting synonyms
def get_synonyms(series):
    word = duden.get(series)
    if word:
        new_entry = word.synonyms
        return new_entry
    else:
        return "entry not found"

all_aus_synonyms = aus_verbs_series.apply(get_synonyms)
print(all_aus_synonyms)


