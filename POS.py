import duden
import pandas as pd
import sqlite3
import re
from datetime import datetime
startTime = datetime.now()

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
    new_entry = new_entry.replace("ö", "oe").replace("ä", "ae").replace("ü", "ue").replace("ß", "sz").replace("Ö", "Oe").replace("Ä", "Ae").replace("Ü", "Ue")
    return new_entry

series_of_words = series_of_words_unparsed.apply(extract_words)

def get_pos(word):
    '''function to get the part of speech of words from duden'''
    while True:
        try:
            w = duden.get(word)
            return w.part_of_speech
            break
        except:
            while True:
                try:
                    ws = duden.search(word)
                    pos = [re.search(r'\((.*?)\)', str(entry)).group(1) for entry in ws]
                    return str(["word not found" if pos == [] else pos])
                    break
                except:
                    return "word not found"

#Applying the function, getting part-of-speech tags
pos_column = series_of_words[130000:135000].apply(get_pos)

#Creating a dataFrame with words and their part-of-speech tag
df_words = pd.DataFrame({"words": series_of_words[130000:135000], "pos": pos_column})

print(df_words)

#Loading the words with pos-tagging to SQL-Database
conn = sqlite3.connect("full_table.db")
c = conn.cursor()
#c.execute("""CREATE TABLE all_words ( words text pos text)""")
df_words.to_sql("all_words", conn, if_exists="append")
conn.commit()

print(pd.read_sql_query("select * from all_words;", conn)) #checking new entries

conn.close()

print(datetime.now() - startTime) #keeping track on processing time




