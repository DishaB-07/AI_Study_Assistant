
import numpy as np
import faiss

def save_index(index, path="vector_db.index"):
    faiss.write_index(index, path)

def create_faiss_index(embeddings):

    embeddings = np.array(embeddings, dtype="float32")

    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index


def search(index, query_embedding, k=3):

    query_embedding = np.array(query_embedding, dtype="float32")

    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    distances, indices = index.search(query_embedding, k)

    return distances, indices

def load_index(path="vector_db.index"):
    index = faiss.read_index(path)
    return index