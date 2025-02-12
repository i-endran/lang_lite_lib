"""
This module provides utility functions for working with various language models (LLMs) and embeddings.
"""

from typing import List, Union
from langchain import hub
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_xai import ChatXAI
from langchain_deepseek import ChatDeepSeek
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from lang_lite.constants import LLMProvider


def get_llm(llm_provider: LLMProvider, model_name: str, temperature: float) -> Union[ChatGoogleGenerativeAI, ChatOpenAI,
        ChatXAI, ChatDeepSeek]:
    """Get the language model based on the type and model name.
    Args:
        llm_provider (LLMType): Type of the language model (GOOGLE, OPENAI, XAI, DEEPSEEK)
        model_name (str): Name of the model (e.g., "gpt-3.5-turbo")
        temperature (float): Temperature for the language model (0.0 to 1.0). Higher values make the model more creative

    Returns:
        Union[ChatGoogleGenerativeAI, ChatOpenAI, ChatXAI, ChatDeepSeek]: Language model instance
    """

    if llm_provider == LLMProvider.GOOGLE:
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
    elif llm_provider == LLMProvider.OPENAI:
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
    elif llm_provider == LLMProvider.XAI:
        return ChatXAI(
            model=model_name,
            temperature=temperature,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
    elif llm_provider == LLMProvider.DEEPSEEK:
        return ChatDeepSeek(
            model=model_name,
            temperature=temperature,
            max_tokens=None,
            timeout=None,
            max_retries=3,
        )
    else:
        raise ValueError(f"Unsupported LLM type: {llm_provider}")


def get_embeddings(llm_provider: LLMProvider, embedding_model_name: str, task_type: str = "semantic_similarity") -> \
        Union[GoogleGenerativeAIEmbeddings, OpenAIEmbeddings]:
    """Get the embeddings based on the type and model name.
    Args:
        llm_provider (LLMType): Type of the language model (GOOGLE, OPENAI)
        embedding_model_name (str): Name of the model (e.g., "text-embedding-ada-002")
        task_type (str): Task type for the embeddings (e.g., "semantic_similarity")
    Returns:
        Union[GoogleGenerativeAIEmbeddings, OpenAIEmbeddings]: The embeddings model instance.
    """

    if llm_provider == LLMProvider.GOOGLE:
        return GoogleGenerativeAIEmbeddings(
            model=embedding_model_name,
            task_type=task_type
        )
    elif llm_provider == LLMProvider.OPENAI:
        return OpenAIEmbeddings(
            model=embedding_model_name
        )
    else:
        raise ValueError(f"Unsupported LLM type: {llm_provider}")


def get_embedded_query(llm_provider: LLMProvider, embedding_model_name, query: str) -> List[float]:
    """Get the embeddings for the query. It is used to retrieve similar documents from the vector store.
    Args:
        llm_provider (LLMType): Type of the language model (GOOGLE, OPENAI)
        embedding_model_name (str): Name of the model (e.g., "text-embedding-ada-002")
        query (str): Query | prompt string
    
    Returns:
        List[float]: The embeddings for the query.
    """
    return get_embeddings(llm_provider, embedding_model_name).embed_query(query)


def query_llm_with_rag(llm, vector_store, role: str, query: str) -> str:
    """Query the language model with RAG (Retrieval-Augmented Generation) pipeline.
    LLM retrieves query embeddings from the vector store and generates the response.
    Args:
        llm: Language model instance
        vector_store: vector store
        role (str): Role for the LLM
        query (str): Query | prompt string
    Returns:
        str: Response from the language model as a string
    """
    chain = _get_qa_chain(llm, vector_store)
    prompt = f"{role}\n\nQuestion:\n{query}"
    return chain.invoke(prompt)


def query_llm_with_context(llm, context: List[Document], role: str, query: str):
    """Query the language model with context.
    LLM generates the response based on the context documents and query.
    These context documents may be retrieved from a search engine, a database, vector store.
    Args:
        llm: Language model instance
        context (List[Document]): Context documents
        role (str): Role for the LLM
        query (str): Query | prompt string
    Returns:
        AIMessage: Response from the language model as an AIMessage object.
    """
    prompt = f"{role}\nContext:\n{_format_docs(context)}\n\nQuestion:\n{query}"
    response = llm.invoke(prompt)
    return response


def query_llm(llm, role: str, query: str):
    """Query the language model with role and query.
    Args:
        llm: Language model instance
        role (str): Role for the LLM
        query (str): Query | prompt string
    Returns:
        AIMessage: Response from the language model as an AIMessage object.
    """

    if not role:
        messages = [("human", query)]
    else:
        messages = [("system", role), ("human", query)]

    response = llm.invoke(messages)
    return response


def _format_docs(docs: List[Document]) -> str:
    return "\n\n".join([doc.page_content for doc in docs])


def _get_qa_chain(llm: ChatGoogleGenerativeAI, vector_store) -> RunnablePassthrough:
    prompt = hub.pull("rlm/rag-prompt")
    retriever = vector_store.as_retriever()
    return (
            {
                "context": retriever | _format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
    )
