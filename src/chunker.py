from langchain_text_splitters import RecursiveCharacterTextSplitter


class Chunker:

    def __init__(self, chunk_size=300, chunk_overlap=50):

        self.text_splitter = RecursiveCharacterTextSplitter(

            chunk_size=chunk_size,

            chunk_overlap=chunk_overlap

        )

    def create_chunks(self, documents):

        chunks = self.text_splitter.split_documents(documents)

        return chunks