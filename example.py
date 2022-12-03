import os
import metapy
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

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
    doc = metapy.index.Document()
    neg_words, pos_words = load_words()

    dir_list = os.listdir("data")
    # print(dir_list)
    movie_ranking = {}
    for file in dir_list:
        with open ('data/' + file) as f:
            doc.content(f.read())
            tokens = clean_up(doc)
            print(tokens)
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
    print(movie_ranking)

#ngram
#idf

# user interactive 1,2,3, idf,..N
# print out rankings based off chose
