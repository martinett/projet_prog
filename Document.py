from datetime import datetime
from gensim.summarization.summarizer import summarize

class Document:
    @staticmethod
    def factory(type, publication):
        #design pattern factory, qui permet de créer des documents de différents types
        if type == "Reddit":
            return RedditDocument(publication.title,
                                  publication.author.name if publication.author != None else "",
                                  datetime.fromtimestamp(publication.created),
                                  publication.url,
                                  publication.selftext.replace("\n", " "),
                                  len(publication.comments))
        elif type == "Arxiv":
            return ArxivDocument(publication["title"].replace("\n", " ").replace("  ", " "),
                                 [auth["name"] for auth in publication["author"]] if isinstance(publication["author"],list) and len(publication["author"]) > 1 else [publication["author"]["name"]],
                                 datetime.strptime(publication["published"], '%Y-%m-%dT%H:%M:%SZ'),
                                 publication["id"],
                                 publication["summary"].replace("\n", " ").replace("  ", " "))
        else:
            raise Exception("Error : wrong type ("+type+")")
    
    def __init__(self, titre, auteur, date, url, texte):
        #titre de la publication
        self.titre = titre
        #auteur de la publication
        self.auteur = auteur
        #date de publication
        self.date = date
        #url où trouver la publication
        self.url = url
        #contenu textuel de la publication
        self.texte = texte
    
    def disp(self):
        #affichage du document
        return "{} ({}) - {}\n{}\n\n{}".format(self.titre,
                                               self.date,
                                               self.auteur,
                                               self.url,
                                               self.texte)
    
    def get_id(self):
        #génère un id de manière déterministe (à partir de l'url)
        id = ""
        for c in self.url:
            if (c >= "a" and c <= "z") or (c >= "A" and c <= "Z") or (c >= "0" and c <= "9") or c == "_":
                id += c
        return id
    
    def get_type(self):
        #à implémenter par les classes filles
        pass
    
    def __str__(self):
        return "["+self.get_type()+"] "+self.titre
    def __repr__(self):
        return self.__str__()
    
    def summary(self):
        #utilise la librairie gensim pour générer un résumé du document
        return summarize(self.texte)


#document correspondant aux publication de reddit.com
class RedditDocument(Document):
    def __init__(self, titre, auteur, date, url, texte, nb_comments):
        super().__init__(titre, auteur, date, url, texte)
        self.nb_comments = nb_comments
    
    def get_type(self):
        return "Reddit"


#document correspondant aux publications de arxiv.org
class ArxivDocument(Document):
    def __init__(self, titre, auteurs, date, url, text):
        super().__init__(titre, auteurs[0], date, url, text)
        self.co_auteurs = auteurs[1:]
        self.nb_auteurs = len(auteurs)
    
    def get_type(self):
        return "Arxiv"
