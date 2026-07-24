from src.data_loader import DataLoader
from src.rag_pipeline import RAGPipeline
from src.evaluator import Evaluator


class Experiment:

    def __init__(self):
        self.loader = DataLoader()

    def run_baseline(self, limit=25, k=3):

        rag = RAGPipeline()
        evaluator = Evaluator()

        questions = self.loader.get_questions(limit)

        print(f"\nRunning Evaluation on {limit} Questions...\n")

        for i, sample in enumerate(questions):

            question = sample["question"]
            ground_truth = sample["answer"]

            prediction, _ = rag.ask(
                question,
                k=k
            )

            evaluator.evaluate(
                question,
                ground_truth,
                prediction
            )

            print(f"{i+1}/{limit} Completed")

        accuracy = evaluator.accuracy()

        print(f"\nAccuracy : {accuracy}%")

        return accuracy, evaluator


    def chunk_size_experiment(self):

        print("\n========== Chunk Size Experiment ==========\n")

        chunk_sizes = [300, 500, 800]

        for size in chunk_sizes:

            print(f"\nChunk Size : {size}")

            print("Please rebuild the vector database using:")
            print(f"Chunker(chunk_size={size}, chunk_overlap=50)")

            input("Press Enter after rebuilding the Vector DB...")

            accuracy, evaluator = self.run_baseline(limit=50)

            evaluator.save_results(f"results_chunk_{size}.csv")

            print(f"Accuracy : {accuracy}%")



    def topk_experiment(self):

        print("\n========== Top-k Experiment ==========\n")

        top_k = [1, 3, 5]

        for k in top_k:

            print(f"\nTop-{k}")

            accuracy, evaluator = self.run_baseline(
                limit=50,
                k=k
            )

            evaluator.save_results(f"results_topk_{k}.csv")

            print(f"Accuracy : {accuracy}%")