import os
import metapy
from nltk.tokenize import word_tokenize
import nltk
import math
from sklearn.feature_extraction.text import TfidfVectorizer
# nltk.download('punkt')
import pandas as pd
from textwrap import wrap

def clean_up(doc, n):
    tok = metapy.analyzers.ICUTokenizer(suppress_tags=True)
    tok = metapy.analyzers.LowercaseFilter(tok)

    # remove common words
    tok = metapy.analyzers.ListFilter(tok, "lemur-stopwords.txt", metapy.analyzers.ListFilter.Type.Reject)
    script = doc.content().replace("\n", "")
    script = doc.content().replace("-", "")

    size = int(len(script) / 3)
    movie_parts = wrap(script, size)

    if n < 4:
        final_script = movie_parts[n-1]
    else: 
        final_script = script

    tok.set_content(final_script)
    doc.content(final_script)
    ana = metapy.analyzers.NGramWordAnalyzer(1, tok)
    grams = ana.analyze(doc)
    return grams

def load_words():
    file = open('negative-words.txt', 'r')
    neg_words = file.read().split()
    file = open('positive-words.txt', 'r')
    pos_words = file.read().split()
    return neg_words, pos_words

def get_score(k, my_dict):
    if k in my_dict:
        return neg_words[k]
    else:
        return 0

if __name__ == '__main__':
    doc = metapy.index.Document()
    neg_words, pos_words = load_words()
    dir_list = os.listdir("data")
    dir_list.remove(".DS_Store")
    # Computes sentiment score based on pos/neg words for each movie
    movie_ranking = {}

    # Takes in user input to select scoring method
    print("Welcome to the Movie Ranking Program!\n")
    print("ABOUT THE PROGRAM:")
    print("In this program we are parsing movie scripts to analyze the sentiment in terms of negative and positive words frequency and assigning a score for each movie.")
    print("\nPROMPT:")
    print("Which part of each movies would you like to parse?\n1) Beginning \n2) Middle \n3) Ending \n4) Whole Movie\n")
    n = input("YOUR ANSWER: ")
    part = ""
    if n == 1:
        part = "beginning part"
    elif n == 2:
        part = "middle part"
    elif n == 3:
        part == "ending part"
    else:
        part == "all parts"

    print("LOADING RESULTS...\n")
    for file in dir_list:
        with open ('data/' + file) as f:
            doc.content(f.read())
            tokens = clean_up(doc, int(n))
            sentiment_dict = dict.fromkeys(tokens.keys(), 0)
            for key in sentiment_dict:
                score = 0
                if key in neg_words:
                    score -= 1
                elif key in pos_words:
                    score += 1
                sentiment_dict[key] = score
            movie_ranking[file] = sum(sentiment_dict.values())
            
    movie_ranking = sorted(movie_ranking.items(), key=lambda x:x[1])
    result_df = pd.DataFrame(movie_ranking, columns =['Movie', "Score"])
    print("ALGORITHM RESULTS:")
    print(result_df)
    print("\nINTERPRETING RESULTS:")
    print("You selected to analyze the", part, "of our movie corpus. The output of our program indicates that out of our movie corpus of", len(dir_list), "movies, we have", result_df['Movie'].iloc[-1],
    "with the highest sentiment score of", result_df['Score'].iloc[-1], "which means the movie is overall the most positive. Inversely, the most movie script with the most negative sentiment according our program is",
    result_df['Movie'].iloc[0], "with a sentiment score of", result_df['Score'].iloc[0], ".")
