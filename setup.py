from setuptools import find_packages, setup

setup(
    name='lang-lite',
    packages=find_packages(include=['lang_lite']),
    version='0.2.1',
    description='Quick prototyping library for LangChain',
    author='narendranseenivasan@gmail.com',
    install_requires=[
        'langchain',
        'langchain-community',
        'tenacity',
        'beautifulsoup4',
        'pandas',
        'langchain-google-genai',
        'langchain-openai',
        'langchain-xai',
        'langchain-deepseek',
        'langchain-text-splitters',
        'faiss-cpu',
    ],  # Add any dependencies required for the library
    setup_requires=[],  # Add any dependencies required for setup
)
