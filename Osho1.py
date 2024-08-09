import streamlit as st
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import openai
from openai import OpenAI
import os
import requests

# Define paths
faiss_index_path = "faiss_index.bin"
chunks_path = "chunks.pkl"
databricks_token = "dapi231e912aa93cedec276065cb8995cce5"
server_endpoint = "https://adb-1769509101973077.17.azuredatabricks.net/serving-endpoints"

# GitHub raw URLs
github_base_url = "https://raw.githubusercontent.com/chiefmanish/Streamlit_pdf/main/"

# Download FAISS index
if not os.path.exists(faiss_index_path):
    response = requests.get(github_base_url + faiss_index_path)
    with open(faiss_index_path, "wb") as f:
        f.write(response.content)

# Download chunks
if not os.path.exists(chunks_path):
    response = requests.get(github_base_url + chunks_path)
    with open(chunks_path, "wb") as f:
        f.write(response.content)

# Load FAISS index
index = faiss.read_index(faiss_index_path)

# Load text chunks
with open(chunks_path, "rb") as f:
    all_chunks = pickle.load(f)

# Load the sentence transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Function to get the most relevant chunks
def get_relevant_chunks(query, index, model, chunks, top_k=10):
    query_embedding = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_embedding, top_k)
    relevant_chunks = [chunks[i] for i in I[0] if i < len(chunks)]
    return relevant_chunks

# Function to get results from the Databricks model
def get_result(databricks_token, server_endpoint, extracted_text, question):
    client = OpenAI(
        api_key=databricks_token,
        base_url=server_endpoint
    )
    response = client.chat.completions.create(
        model="databricks-dbrx-instruct",
        messages=[

            {
                "role": "user",
                "content": f"""
                When answering, refer to yourself as Osho, embodying his wisdom and insight. Think as if you are the one who has written the content and must frame the answer based strictly on the knowledge provided below:
                "{extracted_text}"
                Here is the user question that you have to answer: {question}, except this question don't try to answer or reply that is not this question.
                Respond to questions only if the answer can be found within the content provided. And then build your answer only from the content and while developing your answer from the content you can be creative and try to respond which should be a profound and contemplative explanation, reflecting the depth and insight typical of Osho's teachings. It should not be in a conversational format with a seeker. Instead, provide a reflective and instructive answer, using examples or metaphors as needed, all strictly based on the provided content.
                If the question is not related to the provided content, respond with:
                "Ah, I see where you’re coming from, but this question is not covered in the wisdom at hand. Please ask a question related to the provided content."
                don't try to respond that is not asked by the user.
               
                """
            }

        ],
        temperature=0,
        top_p=0.95,
        max_tokens=500
    )

    return response.choices[0].message.content


# Streamlit UI
st.title("Wisdom Unveiled: Insights from Osho")
query = st.text_input("What question has been dancing in your mind lately? 🤔")

if st.button("Search"):
    if query:
        relevant_chunks = get_relevant_chunks(query, index, model, all_chunks)
        response = get_result(databricks_token, server_endpoint, " ".join(relevant_chunks), query)
        st.write(response)
    else:
        st.write("Please enter a query to search.")
