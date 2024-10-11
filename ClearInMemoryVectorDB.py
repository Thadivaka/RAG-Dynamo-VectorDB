class InMemoryVectorDB:
    def __init__(self):
        self.vectors = {}
        self.metadata = {}

    def add_vector(self, id, vector, metadata=None):
        self.vectors[id] = vector
        if metadata:
            self.metadata[id] = metadata

    def clear(self):
        self.vectors.clear()
        self.metadata.clear()

# Usage
db = InMemoryVectorDB()
# ... add vectors to the db ...
db.clear()