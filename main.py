import os
import metapy
from nltk.tokenize import word_tokenize
import nltk
import math
from sklearn.feature_extraction.text import TfidfVectorizer
# nltk.download('punkt')
import pandas as pd

def clean_up(doc):
    tok = metapy.analyzers.ICUTokenizer(suppress_tags=True)
    tok = metapy.analyzers.LowercaseFilter(tok)

    # remove common words
    tok = metapy.analyzers.ListFilter(tok, "lemur-stopwords.txt", metapy.analyzers.ListFilter.Type.Reject)
    tok.set_content(doc.content())

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
    # Takes in user input to select scoring method
    print("WELCOME TO MOVIE RANKING PROGRAM")
    method = input("Which method would you like to try (grams or tfidf)? ")

    doc = metapy.index.Document()
    neg_words, pos_words = load_words()
    dir_list = os.listdir("data")

    if method == "tfidf":
        dir_list.sort()
        movie_words = {}    # maps each file to script contents
        movie_contents = [] # list of script contents in sorted ascending movie name order
        movie_index = {}    # maps sorted ascending movie name to index (0...number of movies-1)
        i = 0

        # Creating mapping of each file to file contents
        for file in dir_list:
            movie_index[i] = file
            with open ('data/' + file) as f:
                contents = f.read()
                movie_contents.append(contents)
                doc.content(contents)
                tokens = clean_up(doc)
                movie_words[file] = tokens
            i += 1
        # Runs TFIDF and outputs result dataframe with sum scores for each movie
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(movie_contents)
        feature_names = vectorizer.get_feature_names()
        dense = vectors.todense()
        denselist = dense.tolist()

        # Clean up resulting dataframe
        result_df = pd.DataFrame(denselist, columns=feature_names).sum(axis = 1).rename(index=movie_index).to_frame()
        result_df.reset_index(inplace=True)
        result_df.columns = ['Movie', "Score"]
        result_df = result_df.sort_values(by=['Score'], ascending=False)

    elif method == "grams":
        # Computes sentiment score based on pos/neg words for each movie
        movie_ranking = {}
        for file in dir_list:
            with open ('data/' + file) as f:
                doc.content(f.read())
                tokens = clean_up(doc)
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
    print(result_df)
