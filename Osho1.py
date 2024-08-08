import streamlit as st
import faiss
import numpy as np
import pickle
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load the precomputed FAISS index and chunks
faiss_index_path = "/Users/manishkumar/Downloads/faiss_index.bin"
chunks_path = "/Users/manishkumar/Downloads/chunks.pkl"

# Load FAISS index
index = faiss.read_index(faiss_index_path)

# Load text chunks
with open(chunks_path, "rb") as f:
    all_chunks = pickle.load(f)

# Load the GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')

def get_relevant_chunks(query, index, chunks, top_k=5):
    query_embedding = tokenizer.encode(query, return_tensors='pt')
    query_embedding = query_embedding.numpy()
    D, I = index.search(query_embedding, top_k)
    relevant_chunks = [chunks[i] for i in I[0]]
    return relevant_chunks

def get_result(extracted_text, question):
    prompt = f"This is the relevant content: {extracted_text}\nRemember it and this is the question that you have to answer from the content provided: {question}"
    
    # Tokenize and truncate the prompt
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024)
    
    # Generate text
    outputs = gpt2_model.generate(
        inputs['input_ids'],
        max_new_tokens=150,  # Adjust the number of new tokens as needed
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode the output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Streamlit UI
st.title("Document Search App")
query = st.text_input("Enter your query:")

if st.button("Search"):
    if query:
        relevant_chunks = get_relevant_chunks(query, index, all_chunks)
        response = get_result(" ".join(relevant_chunks), query)
        st.write("Response:")
        st.write(response)
    else:
        st.write("Please enter a query to search.")
