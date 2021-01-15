class Auteur:
    def __init__(self, name):
        #nom de l'auteur
        self.name = name
        #nombre de publications que l'auteur a écrit
        self.ndoc = 0
        #collection des publications de l'auteur
        self.production = {}
    
    def add(self, document):
        #si le document n'est pas dans la collection
        if not document in self.production:
            #on l'ajoute
            self.ndoc += 1
            self.production[document.get_id()] = document
    
    def get_id(self):
        #génère l'id de l'auteur de manière déterministe (à partir de son nom)
        id = ""
        for c in self.name:
            if (c >= "a" and c <= "z") or (c >= "A" and c <= "Z") or (c >= "0" and c <= "9") or c == "_":
                id += c
        return id
    
    def __str__(self):
        return self.name
    def __repr__(self):
        return self.__str__()
