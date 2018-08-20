class Cluster:

    def __init__(self, title=""):
        self.title = title
        self.documents = []

    def add_doc(self, document):
        self.documents.append(document)
