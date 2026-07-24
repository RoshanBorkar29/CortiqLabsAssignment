from src.embeddings import VectorStore
from src.generator import Generator


class RAGPipeline:

    def __init__(self):

        self.vector_store = VectorStore()

        self.generator = Generator()

        self.db = self.vector_store.load_vector_store()

    def ask(self, question, k=5):
        """
        Complete RAG Pipeline
        """

        # Retrieve relevant documents
        documents = self.vector_store.retrieve(
            self.db,
            question,
            k=k
        )

        # Generate Answer
        answer = self.generator.generate_answer(
            question,
            documents
        )

        return answer, documents