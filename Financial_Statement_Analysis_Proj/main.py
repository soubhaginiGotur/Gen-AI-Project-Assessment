# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 14:15:10 2024

@author: goturso
"""
import openapi
import os
import time
import streamlit as st
# Streamlit is a framework for building interactive web applications in Python. It allows you to create user interfaces, visualizations, and web apps with minimal code.
from langchain.document_loaders import PyMuPDFLoader
# langchain.document_loaders : provides document loading capabilities,including support for loading PDF files using PyMuPDF.
from langchain.text_splitter import CharacterTextSplitter
# This module provides text splitting capabilities, allowing you to split text into smaller units, such as characters or words.
from langchain.llms import OpenAI
# This module provides Large Language Model (LLM) integration, including support for OpenAI's models.
from langchain.chains import RetrievalQA
# This class is used for question-answering tasks, 
# #where it retrieves relevant information from a knowledge base or corpus.
from langchain.indexes import VectorstoreIndexCreator
# This class is used to create and manage vector-based indexes, 
# #which are useful for semantic search and retrieval tasks.
import tempfile
# This module provides temporary file management capabilities, 
# #allowing you to create and manage temporary files or directories.
from dotenv import load_dotenv
# load envirnment variable from .env
load_dotenv() 

def load_pdf_and_create_index(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_file_path = tmp_file.name

    loader = PyMuPDFLoader(tmp_file_path)
    pages = loader.load_and_split()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(pages)

    index_creator = VectorstoreIndexCreator()
    vectorstore_index = index_creator.from_documents(texts).vectorstore

    return vectorstore_index

def main():
    st.title("Financial Statement Analysis & Compliance Check")

    pdf_file = st.file_uploader("Upload a financial PDF report", type="pdf")

    if pdf_file is not None:
        vectorstore_index = load_pdf_and_create_index(pdf_file)

        llm = OpenAI(temperature=0.2)

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore_index.as_retriever(search_kwargs={"fetch_score": True}),
            return_source_documents=True,
        )

        query1 = st.text_input("Ask a question about the financial metrics report")

        if query1:
            result = qa({"query": query1})
            st.write(f"Answer: {result['result']}")
            st.write(f"Source Documents: {result['source_documents']}")

        query2 = st.text_input("Ask a question about the Compliance Report")

        if query2:
            result = qa({"query": query2})
            st.write(f"Answer: {result['result']}")
            st.write(f"Source Documents: {result['source_documents']}")
            
        query3 = st.text_input("Summary")

        if query3:
            result = qa({"query": query3})
            st.write(f"Answer: {result['result']}")
            st.write(f"Source Documents: {result['source_documents']}")    
            
        query4 = st.text_input("Current financial regulations")

        if query4:
            result = qa({"query": query4})
            st.write(f"Answer: {result['result']}")
            st.write(f"Source Documents: {result['source_documents']}")    
if __name__ == '__main__':
    main()