class Auteur:
    def __init__(self, name):
        self.name = name
        self.ndoc = 0
        self.production = {}
    
    def add(self, document):
        if not document in self.production:
            self.ndoc += 1
            self.production[document.get_id()] = document
    
    def get_id(self):
        id = ""
        for c in self.name:
            if (c >= "a" and c <= "z") or (c >= "A" and c <= "Z") or (c >= "0" and c <= "9") or c == "_":
                id += c
        return id
    
    def __str__(self):
        return self.name
    def __repr__(self):
        return self.__str__()
