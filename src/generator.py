import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

load_dotenv()


class Generator:

    def __init__(self):
        # Pass groq_api_key (or let ChatGroq automatically read GROQ_API_KEY from .env)
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0
        )

    def generate_answer(self, question, documents):
        """
        Generate answer using retrieved documents.
        """
        context = "\n\n".join(
            [doc.page_content for doc in documents]
        )

        prompt = f"""
You are a helpful question answering assistant.

Answer ONLY using the context below.

If the answer is not present, reply:
"I don't know."

Context:
{context}

Question:
{question}

Answer:
"""

        response = self.llm.invoke(
            [HumanMessage(content=prompt)]
        )

        return response.content