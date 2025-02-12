from typing import List
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


class FAISSVectorStoreUtil(object):
    """A class to manage an in-memory vector store for document embeddings using FAISS.
    """

    def __init__(self, embedding_model, docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the vector utility with the given embedding model and documents.
        Args:
            embedding_model: The model used to generate embeddings for the documents.
            docs (List[Document]): A list of documents to be processed.
            chunk_size (int, optional): The size of each chunk when splitting documents. Defaults to 1000.
            chunk_overlap (int, optional): The overlap size between chunks when splitting documents. Defaults to 200.
        Raises:
            ValueError: If the embedding model is empty.
            ValueError: If the documents list is empty.
        """

        self.vector_store = None

        if not embedding_model:
            raise ValueError("Embedding model cannot be empty")
        if not docs:
            raise ValueError("Documents list cannot be empty")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True
        )

        chunks = text_splitter.split_documents(docs)
        self.vector_store = FAISS.from_documents(chunks, embedding_model)

    def get_vector_store(self) -> FAISS:
        """Return the vector store."""
        return self.vector_store

    def retrieve_documents(self, embedded_prompt: List[float], search_k: int) -> List[Document]:
        """
        Retrieve documents from the vector store based on the embedded prompt.
        Args:
            embedded_prompt (List[float]): The embedded prompt vector to search for similar documents.
            search_k (int): The number of top similar documents to retrieve.
        Returns:
            List[Document]: A list of documents that are most similar to the embedded prompt.
        Raises:
            ValueError: If the vector store is empty.
        """
        if not self.vector_store:
            raise ValueError("Vector store cannot be empty")

        return self.vector_store.similarity_search_by_vector(embedded_prompt, k=search_k)

    @staticmethod
    def retrieve_documents_from_store(vector_store: FAISS, embedded_prompt: List[float], search_k: int) -> \
            List[Document]:
        """
        Retrieve documents from the given vector store based on the embedded prompt.
        Args:
            vector_store (FAISS): The vector store to search within.
            embedded_prompt (List[float]): The embedded prompt vector to search for similar documents.
            search_k (int): The number of top similar documents to retrieve.
        Returns:
            List[Document]: A list of documents that are most similar to the embedded prompt.
        Raises:
            ValueError: If the vector store is empty.
        """
        if not vector_store:
            raise ValueError("Vector store cannot be empty")

        return vector_store.similarity_search_by_vector(embedded_prompt, k=search_k)
