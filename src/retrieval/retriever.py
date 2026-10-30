from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfRetriever:

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=10000
        )

        self.chunks = []
        self.chunk_vectors = None

    def index_chunks(self, chunks):
        """
        Convert document chunks into TF-IDF vectors.
        """

        self.chunks = chunks

        texts = [
            chunk["text"]
            for chunk in chunks
            if chunk["text"].strip()
        ]

        if not texts:
            raise ValueError("No valid text chunks provided.")

        self.chunks = [
            chunk for chunk in chunks
            if chunk["text"].strip()
        ]

        self.chunk_vectors = self.vectorizer.fit_transform(texts)

        return len(self.chunks)

    def search(self, query, top_k=5):
        """
        Retrieve the most relevant chunks for a user query.
        """

        if self.chunk_vectors is None:
            raise ValueError("The retriever has not been indexed yet.")

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        query_vector = self.vectorizer.transform([query])

        similarities = cosine_similarity(
            query_vector,
            self.chunk_vectors
        ).flatten()

        ranked_indices = similarities.argsort()[::-1][:top_k]

        results = []

        for index in ranked_indices:

            results.append({
                **self.chunks[index],
                "score": float(similarities[index])
            })

        return results