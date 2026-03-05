from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def create_embeddings(text_chunks):

    texts = [chunk[1] for chunk in text_chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = embeddings.astype("float32")

    return embeddings