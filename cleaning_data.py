import pandas as pd
import sqlite3


#Opening connection to the sql database with all scrapped words + POS and loading as a DataFrame
conn = sqlite3.connect("full_table.db")
c = conn.cursor()
df = pd.read_sql_query("select * from all_words;", conn)


#Removing duplicates if exist
df = df.drop_duplicates(subset = "words", keep = "first")

#Cleaning entries
def remove_brackets(entry):
    '''removing brackts from some entries to provide homogenity of all entries and splitting several POS in two columns'''
    if entry != None:
        new = entry.strip("[]")
        return new

new_pos = df["pos"].apply(remove_brackets)

#Creating a DataFrame with two columns for POS
new_pos = new_pos.str.split(", ", 1, expand = True)
print(new_pos)
def remove_quotes(entry):
    '''removes quotes from some entries for homogenity'''
    if entry != None:
        new = entry.strip("''")
        return new
    else:
        return "no second POS"

#Creating a DataFrame with colmns pos1, pos2, words
full_table = pd.DataFrame(dict(pos1 = new_pos.loc[:, 0].apply(remove_quotes), pos2 = new_pos.loc[:, 1].apply(remove_quotes), words = df["words"])).reset_index()

#Filtering for the not-found-words
not_found_words = full_table.loc[full_table["pos1"] == "word not found"]

#Filtering from not-found-words
full_table_wo_none = full_table.loc[full_table["pos1"] != "word not found"].dropna(axis = 0).reset_index()
full_table_wo_none = full_table_wo_none.drop(columns = "level_0").drop(columns = "index")

#Writing cleaned data in a database, new table cleaned_all_words will be created
full_table_wo_none.to_sql("cleaned_all_words", conn, if_exists="replace")
conn.commit()

#Just in case writing not found words also in a separate table in the database
not_found_words.to_sql("not_found_words", conn, if_exists="replace")
conn.commit()
conn.close()
