from typing import List
from langchain.schema import Document
import lang_lite.model_util as models
import lang_lite.io_util as io_util
from lang_lite.vector_util import FAISSVectorStoreUtil as vs
from lang_lite.constants import LLMProvider


class SimpleRagChain(object):
    def __init__(self, llm_provider: LLMProvider, model_name: str, temperature: float = 0.3):
        self.docs: List[Document] = []
        self.vector_store = None
        self.embedding_model = None
        self.llm = models.get_llm(llm_provider, model_name, temperature)
        self.role = "You are an AI assistant"

    def set_role(self, role: str):
        self.role = role
        return self

    def add_context_urls(self, urls: List[str]):
        if not urls:
            raise ValueError("URLs list cannot be empty")

        docs = io_util.load_documents_from_web(urls)
        self.docs = docs if not self.docs else self.docs + docs
        return self

    def add_context_pdf(self, file_path: str):
        if not file_path:
            raise ValueError("File path cannot be empty")

        docs = io_util.load_documents_from_pdf(file_path)
        self.docs = docs if not self.docs else self.docs + docs
        return self

    def add_context_csv(self, file_path: str):
        if not file_path:
            raise ValueError("File path cannot be empty")

        docs = io_util.load_documents_from_csv(file_path)
        self.docs = docs if not self.docs else self.docs + docs
        return self

    def add_context_docs(self, docs: List[Document]):
        self.docs = docs if not self.docs else self.docs + docs
        return self

    def build_vector_store(self, embedding_model_type: LLMProvider, embedding_model_name: str, chunk_size: int = 1000,
                           chunk_overlap: int = 200):

        if not self.docs:
            raise ValueError("Documents list cannot be empty, add some context documents first")

        self.embedding_model = models.get_embeddings(embedding_model_type, embedding_model_name)
        self.vector_store = vs(self.embedding_model, self.docs, chunk_size, chunk_overlap).get_vector_store()
        return self

    def prompt_without_rag(self, query: str, role: str = None) -> str:
        response = models.query_llm(self.llm, self.role if not role else role, query)
        return response.content

    def prompt_rag(self, query: str, role: str = None) -> str:

        if not self.vector_store:
            raise ValueError("Vector store cannot be empty, add context and build the vector store first")

        response = models.query_llm_with_rag(self.llm, self.vector_store, self.role if not role else role, query)
        return response

    def prompt_with_embedded_query(self, query: str, role: str = None, top_docs_to_search: int = 5) -> str:
        context = vs.retrieve_documents_from_store(self.vector_store,
                                                   self.embedding_model.embed_query(query),
                                                   top_docs_to_search)

        response = models.query_llm_with_context(self.llm, context, self.role if not role else role, query)
        return response.content
