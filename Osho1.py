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
def get_relevant_chunks(query, index, model, chunks, top_k=5):
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
                    As Osho, answer the following question with wisdom and insight drawn from your teachings:

                    Question: "{question}"

                    Content from which you must have to find your answer of the question and frame from it:
                    "{extracted_text}"
                    
                    Your response must be based strictly on the content provided. Don't provide your own insights about the question and answer as if a teacher or guru or poet or spiritual teacher would answer.
                    If the question does not relate to the content provided, respond with:
                    "Please ask a question related to one of the following topics: 'From Darkness to Light', 'From Death to Deathlessness', 'From Ignorance to Innocence', 'From Misery to Enlightenment', 'From Personality to Individuality', 'From Sex to Superconsciousness', 'From the False to the Truth', or 'From Unconsciousness to Consciousness'."

                    Your response should be profound, contemplative, and in line with the spiritual guidance that Osho is known for. And also answer in an interesting way of explaining like a teacher or guru is teaching 
                """
            }

        ],
        temperature=0,
        top_p=0.95,
        max_tokens=500
    )

    return response.choices[0].message.content


# Streamlit UI
st.title("Document Search App")
query = st.text_input("Enter your query:")

if st.button("Search"):
    if query:
        relevant_chunks = get_relevant_chunks(query, index, model, all_chunks)
        response = get_result(databricks_token, server_endpoint, " ".join(relevant_chunks), query)
        st.write(response)
    else:
        st.write("Please enter a query to search.")
