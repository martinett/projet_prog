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
        #nom du corpus
        self.name = name
        #collection des auteurs du corpus
        self.authors = {}
        #dictionnaire permettant l'accès rapide aux auteurs du corpus
        self.id2aut = {}
        #collection des publications du corpus
        self.collection = {}
        #dictionnaire permettant l'accès rapide aux auteurs du corpus
        self.id2doc = {}
        #nombre de publications du corpus
        self.ndoc = 0
        #nombre d'auteur du corpus
        self.naut = 0
        #texte intégral du corpus (textes des publications mis bout-à-bout)
        self.whole_textual_content = ""
        #tableau de fréquence des mots du corpus
        self.frequencies = pd.DataFrame(columns=("term frequency", "document frequency"))
        #matrice de co-occurrences
        self.A = pd.DataFrame()
        #booléen permettant de savoir s'il y a eu de nouvelles données ou non pour effectuer le traitement
        self.update = False
        nltk.download('stopwords', quiet=True)
    
    def _get_reddit_posts_as_documents(self, n, keyword="data"):
        #récupère n publication de reddit.com
        hot_posts = reddit.subreddit(keyword).hot(limit=n)
        posts = []
        for post in hot_posts:
            posts.append(Document.factory("Reddit", post))
        return posts
    
    def _get_arxiv_publications_as_documents(self, n, keyword="data"):
        #récupère n publications de arxiv.org
        url = 'http://export.arxiv.org/api/query?search_query=all:'+keyword+'&start=0&max_results='+str(n)
        data = xmltodict.parse(urllib.request.urlopen(url).read())
        pubs = []
        #si un seul document est requêté, le format est différent
        if n == 1:
            data["feed"]["entry"] = [data["feed"]["entry"]]
        for pub in data["feed"]["entry"]:
            pubs.append(Document.factory("Arxiv", pub))
        return pubs
    
    def add_document(self, doc):
        #ajoute un document aux collections
        docid = doc.get_id()
        #si le document n'est pas encore dans la collection
        if not docid in self.collection:
            #on l'ajoute
            self.update = True
            self.ndoc += 1
            self.collection[docid] = doc
            self._update_id2doc(docid)
            self._add_authors(docid)
            self.whole_textual_content += doc.texte + " " + doc.titre
    
    def download_collection(self, n=10, keyword="data"):
        #télécharge n documents de reddit et arxiv
        docs = []
        n2 = n//2
        n1 = n-n2
        if n1 > 0:
            docs.extend(self._get_reddit_posts_as_documents(n1, keyword=keyword))
        if n2 > 0:
            docs.extend(self._get_arxiv_publications_as_documents(n2, keyword=keyword))
        #une fois téléchargés, on les envoie au pré-traitement
        for doc in docs:
            self.add_document(doc)
    
    def _update_id2doc(self, docid):
        #met à jour le dictionnaire des documents
        self.id2doc[docid] = self.collection[docid].titre
    
    def _add_authors(self, docid):
        #ajoute un auteur aux collections
        if self.collection[docid].auteur != "":
            author = Auteur(self.collection[docid].auteur)
            autid = author.get_id()
            #si l'auteur n'est pas encore dans les collections
            if not autid in self.authors:
                #on l'ajoute
                self.naut += 1
                self.authors[author.get_id()] = author
                self._update_id2aut(autid)
    
    def _update_id2aut(self, autid):
        #met à jour le dictionnaire des auteur
        self.id2aut[autid] = self.authors[autid].name
    
    def get_n_docs(self, n, sort_by="date", reverse=False):
        #récupère les n premiers documents de la collection, ordonnés suivant un citère : la date ou le titre
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
        #enregistre le corpus dans un fichier
        with open(self.name+".corpus", "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
    
    def open(name):
        #charge un corpus depuis un fichier
        with open(name, "rb") as f:
            corpus = pickle.load(f)
        return corpus

    def __repr__(self):
        return "[\n\t- {},\n\t- {}\n\t- {}\n\t...\n]\n({} docs)".format(*self.get_n_docs(3), self.ndoc)
    
    def search(self, keyword):
        #cherche un mot dans le texte intégral
        return len(re.findall(keyword, self.whole_textual_content))
    
    def concorde(self, keyword, nchar):
        #récupère les occurrences d'un mot ainsi que les n caractères qui le précèdent et le suivent
        n = len(keyword)
        matches = re.findall(".{0,"+str(nchar)+"}"+keyword+".{0,"+str(nchar)+"}", self.whole_textual_content)
        df = pd.DataFrame()
        #on parcourt chaque occurrence
        for match in matches:
            #et on récupère les occurrence + les caractères
            i = match.index(keyword)
            j = i+n
            df = df.append(pd.DataFrame(((match[:i], match[i:j], match[j:]),)), ignore_index=True)
        return df
    
    def nettoyer_texte(self, texte):
        #nettoie le texte en ne récupérant que les mots
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
        #calcule et stocke en mémoire les fréquences des mots du corpus
        self.update = False
        vocab = {}
        #on parcourt les documents
        for docid in self.collection:
            #on récupère le texte qu'on split par mots
            texte = self.nettoyer_texte(self.collection[docid].texte)
            encountered = set()
            words = list(texte.split(" "))
            while "" in words:
                words.remove("")
            #on parcours les mots du document
            for word in words:
                #si on n'a pas encore vu ce mot
                if not word in vocab:
                    #on l'ajoute à notre vocabulaire
                    vocab[word] = [0, 0]
                #si on n'a pas encore vu ce mot dans ce document
                if not word in encountered:
                    #on l'ajoute au vocabulaire connu pour ce document
                    encountered.add(word)
                    vocab[word][1] += 1
                vocab[word][0] += 1
                #si le mot n'est pas encore dans la matrice d'occurrences
                if word not in self.A:
                    #on l'ajoute
                    self.A[word] = 0
                    self.A.loc[word] = 0
            self.A.loc[words,words]+=1
            np.fill_diagonal(self.A.values, 0)
        #on enregistre le vocabulaire rencontré dans un tableau
        df = pd.DataFrame.from_dict(vocab, orient="index", columns=("term frequency", "document frequency"))
        df = df.sort_values("term frequency", ascending=False)
        #qu'on stocke dans un attribut
        self.frequencies = df
    
    def stats(self, n):
        #affiche des statistiques sur les n mots les plus fréquents
        if self.update:
            #on ne reconstruit le vocabulaire que s'il y a eu de nouvelles données
            self.words_frequency()
        print("Number of words :", len(self.frequencies))
        print(self.frequencies.index[:n])
    
    def most_frequent_words(self, n):
        #on récupère les n mots les plus fréquents
        if self.update:
            #on ne reconstruit le vocabulaire que s'il y a eu de nouvelles données
            self.words_frequency()
        return self.frequencies.index[:n].values
    
    def get_frequencies(self, word):
        #on récupère la fréquence d'un mot
        return self.frequencies[self.frequencies.index == word]
    
    def is_stop_words(self, word):
        #vérifie si le mot est un stopword ou non
        return word in stopwords.words('english')

    def get_adjacency_matrix(self):
        #récupère la matrice de co-occurrence
        if self.update:
            #on ne reconstruit le vocabulaire que s'il y a eu de nouvelles données
            self.words_frequency()
        return self.A
        