import os
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

if __name__ == '__main__':
    print("Loading Document")
    loader = TextLoader("medium_article.txt",  encoding = 'UTF-8')
    document = loader.load()

    print("Splitting")
    text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts=text_splitter.split_documents(document)
    
    embeddings = OpenAIEmbeddings()

    print("Ingesting in Vector Store")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ['INDEX_NAME'])
    print("Finished Ingestion")