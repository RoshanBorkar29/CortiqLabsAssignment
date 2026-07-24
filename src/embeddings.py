from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


class VectorStore:

    def __init__(self):

        self.embedding_model = HuggingFaceEmbeddings(

            model_name="sentence-transformers/all-MiniLM-L6-v2"

        )

    def create_vector_store(self, chunks):

        return FAISS.from_documents(

            chunks,

            self.embedding_model

        )

    def save_vector_store(self, vector_db):

        vector_db.save_local("vector_db")

    def load_vector_store(self):

        return FAISS.load_local(

            "vector_db",

            self.embedding_model,

            allow_dangerous_deserialization=True

        )

    def retrieve(self, vector_db, query, k=5):

        return vector_db.similarity_search(

            query,

            k=k

        )