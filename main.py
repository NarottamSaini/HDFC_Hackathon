import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
# from langchain_community.embeddings import OpenAIEmbeddings
# from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
# from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

st.title("HDFC Chatbot for FAQ !!!")

file_path = ".\\notebooks\\vector_index.pkl"

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500, model = "gpt-3.5-turbo-instruct")  ## model = "gpt-3.5-turbo", , model= "gpt-3.5-turbo-instruct", 'gpt-4o-mini'


query = main_placeholder.text_input("Question: ")
print("query : ", query)
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            main_placeholder.text("loading Embedding Vector ...✅✅✅")
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Query")
            st.write(query)
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)




