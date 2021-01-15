import praw
import urllib
import xmltodict
import pickle
import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords

from Document import Document
from Auteur import Auteur

# https://www.reddit.com/prefs/apps
reddit = praw.Reddit(client_id='C9EzqOg-Ie1ptw',
                     client_secret='SsYsi8UCF5bUUG9dYuu_QRea2Rw',
                     user_agent='scrapping')


class Corpus:
    def __init__(self, name):
        self.name = name
        self.authors = {}
        self.id2aut = {}
        self.collection = {}
        self.id2doc = {}
        self.ndoc = 0
        self.naut = 0
        self.whole_textual_content = ""
        self.frequencies = pd.DataFrame(columns=("term frequency", "document frequency"))
        self.A = pd.DataFrame()
        self.update = False
        nltk.download('stopwords', quiet=True)
    
    def _get_reddit_posts_as_documents(self, n, keyword="data"):
        hot_posts = reddit.subreddit(keyword).hot(limit=n)
        posts = []
        for post in hot_posts:
            posts.append(Document.factory("Reddit", post))
        return posts
    
    def _get_arxiv_publications_as_documents(self, n, keyword="data"):
        url = 'http://export.arxiv.org/api/query?search_query=all:'+keyword+'&start=0&max_results='+str(n)
        data = xmltodict.parse(urllib.request.urlopen(url).read())
        pubs = []
        if n == 1:
            data["feed"]["entry"] = [data["feed"]["entry"]]
        for pub in data["feed"]["entry"]:
            pubs.append(Document.factory("Arxiv", pub))
        return pubs
    
    def add_document(self, doc):
        docid = doc.get_id()
        if not docid in self.collection:
            self.update = True
            self.ndoc += 1
            self.collection[docid] = doc
            self._update_id2doc(docid)
            self._add_authors(docid)
            self.whole_textual_content += doc.texte + " " + doc.titre
    
    def download_collection(self, n=10, keyword="data"):
        docs = []
        n2 = n//2
        n1 = n-n2
        if n1 > 0:
            docs.extend(self._get_reddit_posts_as_documents(n1, keyword=keyword))
        if n2 > 0:
            docs.extend(self._get_arxiv_publications_as_documents(n2, keyword=keyword))
        for doc in docs:
            self.add_document(doc)
    
    def _update_id2doc(self, docid):
        self.id2doc[docid] = self.collection[docid].titre
    
    def _add_authors(self, docid):
        if self.collection[docid].auteur != "":
            author = Auteur(self.collection[docid].auteur)
            autid = author.get_id()
            if not autid in self.authors:
                self.naut += 1
                self.authors[author.get_id()] = author
                self._update_id2aut(autid)
    
    def _update_id2aut(self, autid):
        self.id2aut[autid] = self.authors[autid].name
    
    def get_n_docs(self, n, sort_by="date", reverse=False):
        docs = list(self.collection.values())
        if sort_by == "date":
            key = lambda doc : doc.date
        elif sort_by == "title":
            key = lambda doc : doc.titre
        else:
            raise Exception("{} is not a key that can be used to sort documents.\nDocuments can be sorted by date or title.")
        docs = sorted(docs, key=key, reverse=reverse)
        return docs[:n]
    
    def save(self):
        with open(self.name+".corpus", "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
    
    def open(name):
        with open(name, "rb") as f:
            corpus = pickle.load(f)
        return corpus

    def __repr__(self):
        return "[\n\t- {},\n\t- {}\n\t- {}\n\t...\n]\n({} docs)".format(*self.get_n_docs(3), self.ndoc)
    
    def search(self, keyword):
        return len(re.findall(keyword, self.whole_textual_content))
    
    def concorde(self, keyword, nchar):
        n = len(keyword)
        matches = re.findall(".{0,"+str(nchar)+"}"+keyword+".{0,"+str(nchar)+"}", self.whole_textual_content)
        df = pd.DataFrame()
        for match in matches:
            i = match.index(keyword)
            j = i+n
            df = df.append(pd.DataFrame(((match[:i], match[i:j], match[j:]),)), ignore_index=True)
        return df
    
    def nettoyer_texte(self, texte):
        texte = str.lower(texte)
        texte = re.sub("(\W|\d|_)+", " ", texte)
        while "\n" in texte:
            texte = texte.replace("\n", " ")
        while "  " in texte:
            texte = texte.replace("  ", " ")
        while len(texte) > 0 and texte[0] == " ":
            texte = texte[1:]
        while len(texte) > 0 and texte[-1] == " ":
            texte = texte[:-1]
        return texte
    
    def words_frequency(self):
        self.update = False
        vocab = {}
        for docid in self.collection:
            texte = self.nettoyer_texte(self.collection[docid].texte)
            encountered = set()
            words = list(texte.split(" "))
            while "" in words:
                words.remove("")
            for word in words:
                if not word in vocab:
                    vocab[word] = [0, 0]
                if not word in encountered:
                    encountered.add(word)
                    vocab[word][1] += 1
                vocab[word][0] += 1
                if word not in self.A:
                    self.A[word] = 0
                    self.A.loc[word] = 0
            self.A.loc[words,words]+=1
            np.fill_diagonal(self.A.values, 0)
        df = pd.DataFrame.from_dict(vocab, orient="index", columns=("term frequency", "document frequency"))
        df = df.sort_values("term frequency", ascending=False)
        self.frequencies = df
    
    def stats(self, n):
        if self.update:
            self.words_frequency()
        print("Number of words :", len(self.frequencies))
        print(self.frequencies.index[:n])
    
    def most_frequent_words(self, n):
        if self.update:
            self.words_frequency()
        return self.frequencies.index[:n].values
    
    def get_frequencies(self, word):
        return self.frequencies[self.frequencies.index == word]
    
    def is_stop_words(self, word):
        return word in stopwords.words('english')

    def get_adjacency_matrix(self):
        if self.update:
            self.words_frequency()
        return self.A
        