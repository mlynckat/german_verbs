import numpy as np
import pandas as pd
import sqlite3
import duden
import re
from datetime import datetime
startTime = datetime.now()

conn = sqlite3.connect("full_table.db")
c = conn.cursor()
df = pd.read_sql_query("select * from all_words;", conn)

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

not_found_words = full_table.loc[full_table["pos1"] == "word not found"].reset_index()["words"]

def get_pos(word):
    '''function to get the part of speech of words'''
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


pos_column = not_found_words[1000:5000].apply(get_pos)
df_words = pd.DataFrame({"words": not_found_words[1000:5000], "pos": pos_column})

#c.execute("""CREATE TABLE not_found ( index integer words text pos text)""")

df_words.to_sql("not_found", conn, if_exists="replace")
conn.commit()

print(pd.read_sql_query("select * from not_found;", conn))
conn.close()
print(datetime.now() - startTime)
