import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

if __name__ == '__main__':
    print("Loading Document")
    loader = PyPDFLoader("react_ppr.pdf")
    document = loader.load()
    text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs=text_splitter.split_documents(document)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index_react")

    new_vectorstore = FAISS.load_local(
        "faiss_index_react", embeddings, allow_dangerous_deserialization=True #Not recommended in Prod Systemss
    )

    llm = ChatOpenAI()
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    #Augment Prompt
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain=create_retrieval_chain(
        retriever=new_vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )
    query = "Give me the gist of ReAct in 3 sentences?"
    result = retrieval_chain.invoke(input={"input": query})

    print(result)