import pandas as pd


class Evaluator:

    def __init__(self):
        self.results = []

    def evaluate(self, question, ground_truth, prediction):

        prediction = prediction.strip().lower()
        ground_truth = ground_truth.strip().lower()

        is_correct = ground_truth in prediction

        self.results.append({

            "Question": question,

            "Ground Truth": ground_truth,

            "Prediction": prediction,

            "Correct": is_correct

        })

    def save_results(self, filename="results.csv"):

        df = pd.DataFrame(self.results)

        df.to_csv(filename, index=False)

        print(f"Results saved to {filename}")

    def accuracy(self):

        if len(self.results) == 0:
            return 0

        correct = sum(result["Correct"] for result in self.results)

        return round((correct / len(self.results)) * 100, 2)