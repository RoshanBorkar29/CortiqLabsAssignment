import os

from src.data_loader import DataLoader
from src.chunker import Chunker
from src.embeddings import VectorStore
from src.rag_pipeline import RAGPipeline


def build_vector_db():

    print("Creating Vector Database...")

    loader = DataLoader()

    documents = loader.get_documents(limit=1000)

    chunker = Chunker()

    chunks = chunker.create_chunks(documents)

    store = VectorStore()

    db = store.create_vector_store(chunks)

    store.save_vector_store(db)

    print("Vector Database Created Successfully!")


def main():

    if not os.path.exists("vector_db"):

        build_vector_db()

    rag = RAGPipeline()

    while True:

        question = input("\nAsk a Question (type 'exit' to quit): ")

        if question.lower() == "exit":
            break

        answer, docs = rag.ask(question)

        print("\nAnswer:\n")
        print(docs)

        print(answer)


if __name__ == "__main__":
    main()