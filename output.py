from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS

# Constants (replace with your actual values if different)
GOOGLE = "google"  # Assuming this is how you identify Google in your constants
DEFAULT_GOOGLE_LLM = "models/gemini-1.5-pro-latest"  # Replace with the actual model name
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
response = qa_chain({"query": query})

print("\nResponse from LLM with RAG: ")
print(response["result"])

print("\nResponse from LLM with RAG: ")
print(response["result"])


# 2. LLM chain with RAG and embedded input query (using retriever's search)
# The previous RAG setup already created the vectorstore and embeddings.  We reuse them.

# Run chain - using retriever directly to get context, then LLM to generate answer
retrieved_docs = retriever.get_relevant_documents(query)
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

print("\n\nResponse from LLM with RAG and embedded query: ")
print(response.content)


# 3. LLM chain without RAG
llm = ChatGoogleGenerativeAI(model=DEFAULT_GOOGLE_LLM, temperature=0.3)
response = llm.invoke("What is the latest news?")

print("\n\nResponse from LLM without RAG: ")
print(response.content)