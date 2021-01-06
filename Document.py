from datetime import datetime
from gensim.summarization.summarizer import summarize

class Document:
    @staticmethod
    def factory(type, publication):
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
        self.titre = titre
        self.auteur = auteur
        self.date = date
        self.url = url
        self.texte = texte
    
    def disp(self):
        return "{} ({}) - {}\n{}\n\n{}".format(self.titre,
                                               self.date,
                                               self.auteur,
                                               self.url,
                                               self.texte)
    
    def get_id(self):
        id = ""
        for c in self.url:
            if (c >= "a" and c <= "z") or (c >= "A" and c <= "Z") or (c >= "0" and c <= "9") or c == "_":
                id += c
        return id
    
    def get_type(self):
        pass
    
    def __str__(self):
        return "["+self.get_type()+"] "+self.titre
    def __repr__(self):
        return self.__str__()
    
    def summary(self):
        return summarize(self.texte)


class RedditDocument(Document):
    def __init__(self, titre, auteur, date, url, texte, nb_comments):
        super().__init__(titre, auteur, date, url, texte)
        self.nb_comments = nb_comments
    
    def get_type(self):
        return "Reddit"


class ArxivDocument(Document):
    def __init__(self, titre, auteurs, date, url, text):
        super().__init__(titre, auteurs[0], date, url, text)
        self.co_auteurs = auteurs[1:]
        self.nb_auteurs = len(auteurs)
    
    def get_type(self):
        return "Arxiv"
