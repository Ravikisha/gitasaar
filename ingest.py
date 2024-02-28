"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle

# from langchain.document_loaders import ReadTheDocsLoader
# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores.faiss import FAISS
from langchain_community.vectorstores import FAISS
# from langchain.document_loaders import UnstructuredFileLoader
# from langchain.document_loaders import PyPDFLoader
# import json
import pandas as pd
# from langchain.document_loaders import DataFrameLoader
from langchain_community.document_loaders import DataFrameLoader
import sqlite3
from langchain.text_splitter import RecursiveCharacterTextSplitter




def ingest_docs(vector):
    """Get documents from web pages."""
    
    # loader = PyPDFLoader(docs)
    cnx = sqlite3.connect('finalGita.db')
    df = pd.read_sql_query("SELECT * FROM gita", cnx)
    df['text'] = df.apply(lambda x: f''' Synonyms: {x['Synonyms']} Translation: {x['Translation']} Purport: {x['Purport']} Purport2: {x['Purport2']} Purport3: {x['Purport3']} Purport4: {x['Purport4']} Purport5: {x['Purport5']} Purport6: {x['Purport6']}''', axis=1)
    
    df.drop(['Devanagari','verseText','Synonyms','Translation','Purport','Purport2','Purport3','Purport4','Purport5','Purport6'], axis=1, inplace=True)
    cnx.close()
    loader = DataFrameLoader(df)
    
    
    raw_documents = loader.load()
    # print("Raw Document: ",raw_documents)
    
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=1000,
    #     chunk_overlap=200,
    # )
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)
    
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save vectorstore
    # with open(vector, "wb") as f:
    #     pickle.dump(vectorstore, f)
    vectorstore.save_local("gita_index")

if __name__ == "__main__":
    ingest_docs('vectorstore7.pkl')

