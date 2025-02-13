"""
This module provides utility functions for loading documents from the web, pdf, csv in langchain.
"""

from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain.schema import Document
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, CSVLoader, DirectoryLoader, TextLoader


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def load_documents_from_web(urls: List[str]) -> List[Document]:
    """Load data as documents from the web using the given URLs
    
    Args:
        urls (List[str]): List of URLs to load data from.

    Returns:
        List[Document]: List of documents loaded from the web as langchain document.
    """

    if not urls:
        raise ValueError("URLs list cannot be empty")

    loader = WebBaseLoader(urls)
    return loader.load()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def load_documents_from_pdf(file_path: str) -> List[Document]:
    """Load data as documents from a PDF file using the given file path
    
    Args:
        file_path (str): Path to the PDF file to load data from.

    Returns:
        List[Document]: List of documents loaded from the PDF file as langchain document.
    """

    if not file_path:
        raise ValueError("File path cannot be empty")

    loader = PyPDFLoader(file_path)
    return loader.load()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def load_documents_from_csv(file_path: str) -> List[Document]:
    """Load data as documents from a CSV file using the given file path
    
    Args:
        file_path (str): Path to the CSV file to load data from.

    Returns:
        List[Document]: List of documents loaded from the CSV file as langchain document.
    """

    if not file_path:
        raise ValueError("File path cannot be empty")

    loader = CSVLoader(file_path)
    return loader.load()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def load_txt_documents_from_directory(file_path: str, recursive: bool = False, file_pattern: str | None = None) -> List[Document]:
    """Load text files as documents from a directory using the given file path

    Args:
        file_path (str): Path to the directory to load data from.
        recursive: (bool): Flag to indicate if the loader should load files recursively from the directory.
        file_pattern: (str): Glob pattern to filter files in the directory.

    Returns:
        List[Document]: List of documents loaded from the directory as langchain document.
    """

    if not file_path:
        raise ValueError("File path cannot be empty")

    if file_pattern is None:
        loader = DirectoryLoader(
            file_path,
            loader_cls=TextLoader,
            recursive=recursive)
    else:
        loader = DirectoryLoader(
            file_path,
            glob=file_pattern,  #"**/*.py",  # Adjust the glob pattern as needed
            loader_cls=TextLoader,
            recursive=recursive)

    return loader.load()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def load_documents_from_txt(file_path: str) -> List[Document]:
    """Load text files as documents from a directory using the given file path

    Args:
        file_path (str): Path to the directory to load data from.

    Returns:
        List[Document]: List of documents loaded from the directory as langchain document.
    """

    if not file_path:
        raise ValueError("File path cannot be empty")

    loader = TextLoader(file_path)
    return loader.load()
