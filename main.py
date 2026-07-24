import os

from src.data_loader import DataLoader
from src.chunker import Chunker
from src.embeddings import VectorStore
from src.rag_pipeline import RAGPipeline
from src.experiments import Experiment


def build_vector_db(chunk_size=300):

    print(f"\nCreating Vector Database (Chunk Size = {chunk_size})...")

    loader = DataLoader()

    documents = loader.get_documents(limit=1000)

    chunker = Chunker(
        chunk_size=chunk_size,
        chunk_overlap=50
    )

    chunks = chunker.create_chunks(documents)

    store = VectorStore()

    db = store.create_vector_store(chunks)

    store.save_vector_store(db)

    print("Vector Database Created Successfully!")


def interactive_mode():

    rag = RAGPipeline()

    while True:

        question = input("\nAsk a Question (type 'exit' to quit): ")

        if question.lower() == "exit":
            break

        answer, docs = rag.ask(question)

        print("\nRetrieved Documents:\n")

        for i, doc in enumerate(docs):
            print(f"\n---------- Document {i+1} ----------")
            print(doc.page_content[:300])

        print("\nAnswer:\n")
        print(answer)


def evaluation_mode(k=3):

    experiment = Experiment()

    experiment.run_baseline(limit=20, k=k)


def main():

    print("\n========== RAG Research Assignment ==========")
    print("1. Interactive RAG")
    print("2. Baseline Evaluation")
    print("3. Chunk Size Experiment")
    print("4. Top-k Experiment")
    print("5. Exit")

    choice = input("\nEnter your choice: ")

    if choice == "1":

        if not os.path.exists("vector_db"):
            build_vector_db()

        interactive_mode()

    elif choice == "2":

        if not os.path.exists("vector_db"):
            build_vector_db()

        evaluation_mode()

    elif choice == "3":

        chunk_size = int(input("Enter Chunk Size (300/500/800): "))

        if os.path.exists("vector_db"):
            import shutil
            shutil.rmtree("vector_db")

        build_vector_db(chunk_size)

        evaluation_mode()

    elif choice == "4":

        if not os.path.exists("vector_db"):
            build_vector_db()

        k = int(input("Enter Top-k (1/3/5): "))

        evaluation_mode(k)

    elif choice == "5":
        print("Goodbye!")

    else:
        print("Invalid Choice")


if __name__ == "__main__":
    main()