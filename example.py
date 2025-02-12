from lang_lite import lite_chain
from lang_lite.constants import LLMProvider
from lang_lite import constants

# LLm chain with RAG (Retrieval Augmented Generation).
# In this chain LLM searches the vector store for the most relevant context and generates the response.
response = (lite_chain
            .SimpleRagChain(LLMProvider.GOOGLE, constants.get_default_llm(LLMProvider.GOOGLE))
            .add_context_urls(["https://www.bbc.com/news/world"])
            .build_vector_store(LLMProvider.GOOGLE, constants.get_default_embedding(LLMProvider.GOOGLE))
            .prompt_rag("What is the latest news?"))

print("\n\nResponse from LLM with RAG: ")
print(response)

# LLm chain with RAG (Retrieval Augmented Generation) and embedded input query.
# In this chain the input query is embedded and used to search the vector store for the most relevant context.
# The response is generated based on the context and the input query by the LLM.
response = (lite_chain
            .SimpleRagChain(LLMProvider.GOOGLE, constants.get_default_llm(LLMProvider.GOOGLE))
            .add_context_urls(["https://www.bbc.com/news/world"])
            .build_vector_store(LLMProvider.GOOGLE, constants.get_default_embedding(LLMProvider.GOOGLE))
            .prompt_with_embedded_query("What is the latest news?"))

print("\n\nResponse from LLM with RAG and embedded query: ")
print(response)

# LLm chain without RAG (Retrieval Augmented Generation).
# In this chain the LLM generates the response directly without searching the vector store for context.
response = (lite_chain
            .SimpleRagChain(LLMProvider.GOOGLE, constants.get_default_llm(LLMProvider.GOOGLE))
            .prompt_without_rag("What is the latest news?"))

print("\n\nResponse from LLM without RAG: ")
print(response)