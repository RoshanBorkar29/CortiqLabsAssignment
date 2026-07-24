from datasets import load_dataset
from langchain_core.documents import Document


class DataLoader:

    def __init__(self):
        print("Loading SQuAD v2 Dataset...")
        self.dataset = load_dataset("rajpurkar/squad_v2")

    def get_documents(self, limit=1000):

        documents = []
        seen = set()

        for sample in self.dataset["train"]:

            context = sample["context"]

            if context not in seen:

                seen.add(context)

                documents.append(
                    Document(
                        page_content=context,
                        metadata={
                            "title": sample["title"]
                        }
                    )
                )

            if len(documents) >= limit:
                break

        return documents

    def get_questions(self, limit=100):

        validation = self.dataset["validation"].shuffle(seed=42)

        questions = []

        for sample in validation.select(range(limit)):

            answer = ""

            if sample["answers"]["text"]:
                answer = sample["answers"]["text"][0]

            questions.append({

                "question": sample["question"],

                "answer": answer,

                "context": sample["context"]

            })

        return questions