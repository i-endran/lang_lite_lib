from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

"""
`output.py` is the converted code from lang_lite library in `example.py` to LangChain code.
Some of the outdated code has been updated to the latest LangChain code for reference.
"""

# Constants (replace with your actual values if different)
GOOGLE = "google"  # Assuming this is how you identify Google in your constants
DEFAULT_GOOGLE_LLM = "gemini-2.0-flash"  # Replace with the actual model name
DEFAULT_GOOGLE_EMBEDDING = "models/embedding-001"  # Replace with the actual embedding model name

# 1. LLM chain with RAG
# Load documents
loader = WebBaseLoader(["https://www.bbc.com/news/world"])
documents = loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = GoogleGenerativeAIEmbeddings(model=DEFAULT_GOOGLE_EMBEDDING)
db = FAISS.from_documents(texts, embeddings)

# Create retriever
retriever = db.as_retriever()

# Create prompt template
prompt_template = """You are an AI assistant. Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Create chain
llm = ChatGoogleGenerativeAI(model=DEFAULT_GOOGLE_LLM, temperature=0.3)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=False
)

# Run chain
query = "What is the latest news?"
response = qa_chain.invoke(f"query: {query}") #invoke must be used instead of qa_chain({})

print("\nResponse from LLM with RAG: ")
print(response["result"])

# only one print statement is in the example.py file
# print("\nResponse from LLM with RAG: ")
# print(response["result"])


# 2. LLM chain with RAG and embedded input query (using retriever's search)
# The previous RAG setup already created the vectorstore and embeddings.  We reuse them.

# Run chain - using vector store directly to get context, then LLM to generate answer
retrieved_docs = db.similarity_search_by_vector(embeddings.embed_query(query), k=5) # use similarity_search_by_vector instead of retrieve_documents
context = "\n".join([doc.page_content for doc in retrieved_docs])

prompt = f"""You are an AI assistant. Use the following context to answer the question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {query}
Answer:"""

llm = ChatGoogleGenerativeAI(model=DEFAULT_GOOGLE_LLM, temperature=0.3)
response = llm.invoke(prompt)

print("\n\nResponse from LLM with RAG and embedded query: ")
print(response.content)

# only one print statement is in the example.py file
# print("\n\nResponse from LLM with RAG and embedded query: ")
# print(response.content)


# 3. LLM chain without RAG
llm = ChatGoogleGenerativeAI(model=DEFAULT_GOOGLE_LLM, temperature=0.3)
response = llm.invoke("What is the latest news?")

print("\n\nResponse from LLM without RAG: ")
print(response.content)