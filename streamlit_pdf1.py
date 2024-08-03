import pdfplumber  # For extracting text from PDF files
import requests  # For making HTTP requests to the Databricks API
import streamlit as st  # For creating the web application
import pandas as pd 
import openai
from openai import OpenAI

def extract_text_from_pdf(pdf_path):
    page_content = ""   
    with pdfplumber.open(pdf_path) as pdf:
        # Iterate through each page of the PDF
        for i, page in enumerate(pdf.pages):
            # Extract text from the current page
            page_text = page.extract_text()
            # Store the text in the dictionary with the page number as the key
            page_content = page_content + page_text 
    return page_content

def get_pdf_contents(pdf1, pdf2):
    # Extract text from the PDF and get the page numbers
    extracted_text1 = extract_text_from_pdf(pdf1)
    extracted_text2 = extract_text_from_pdf(pdf2)
    return extracted_text1, extracted_text2

# Function to call the Databricks API
# def call_databricks_api(pdf1_content, pdf2_content, question):
#     DATABRICKS_URL = "https://adb-1820764323029816.16.azuredatabricks.net/api/2.0/jobs/run-now"
#     TOKEN = "dapi81add3935cf4014069dce8a94283ac37"
#     JOB_ID = "171331411506288"

#     headers = {
#         "Authorization": f"Bearer {TOKEN}"
#     }
#     payload = {
#         "job_id": JOB_ID,
#         "notebook_params": {
#             "pdf1": pdf1_content,
#             "pdf2": pdf2_content,
#             "question": question
#         }
#     }

#     response = requests.post(DATABRICKS_URL, headers=headers, json=payload)
#     return response

def get_result(databricks_token, server_endpoint, extracted_text1, extracted_text2, question):

    user_question = question
    client = OpenAI(
        api_key=databricks_token,
        base_url=server_endpoint
    )

    response = client.chat.completions.create(
        model="databricks-dbrx-instruct",
        messages=[
            {
                "role": "user",
                "content": "This is the content of the first pdf :"+extracted_text1+"\nRemember it"
            }
        ],
        temperature=0,
        top_p=0.95,
        max_tokens=500
    )

    response1 = response.choices[0].message.content

    response = client.chat.completions.create(
        model="databricks-dbrx-instruct",
        messages=[
            {
                "role": "user",
                "content": "This is the content of the first pdf :"+extracted_text1+"\nRemember it"
            },
            {
                "role": "assistant",
                "content":response1
            },
            {
                "role": "user",
                "content": "This the content of second pdf:\n"+extracted_text2+"\n\nRemember it."
            }
        ],
        temperature=0,
        top_p=0.95,
        max_tokens=500
    )
    response2 = response.choices[0].message.content

    response = client.chat.completions.create(
        model="databricks-dbrx-instruct",
        messages=[
            {
                "role": "user",
                "content": "This is the content of the first pdf :"+extracted_text1+"\nRemember it"
            },
            {
                "role": "assistant",
                "content":response1
            },
            {
                "role": "user",
                "content": "This the content of second pdf:\n"+extracted_text2+"\n\nRemember it."
            },
            {
                "role": "assistant",
                "content":response2
            },
            {
                "role": "user",
                "content": "Respond to questions only if the answer can be found within the content of these PDFs. If the answer is not present in the PDFs, kindly respond with - The answer is not in the provided PDFs. If question is realted to you then say - Please ask question about the uploaded pdf's. Here is the question that you have to answer -" + user_question
            }
        ],
        temperature=0,
        top_p=0.95,
        max_tokens=500
    )
    return response.choices[0].message.content

def main():
    
    databricks_token = "dapi81add3935cf4014069dce8a94283ac37"
    server_endpoint = "https://adb-1820764323029816.16.azuredatabricks.net/serving-endpoints"
    # Streamlit app layout
    st.title("PDF Question Answering App")

    uploaded_file1 = st.file_uploader("Upload PDF 1", type="pdf")
    uploaded_file2 = st.file_uploader("Upload PDF 2", type="pdf")
    question = st.text_input("Enter your question")

    if st.button("Get Answer"):
        if uploaded_file1 is not None and uploaded_file2 is not None and question:
            # Read PDF content
            content1, content2 = get_pdf_contents(uploaded_file1, uploaded_file2)
            # Make a request to Databricks API (assuming a REST API endpoint is available)
            response = get_result(databricks_token, server_endpoint, content1, content2, question)
            if response:
                answer = response
                st.write(f"Answer: {answer}")
            else:
                st.write("Error: Unable to get the answer from Databricks.")
        else:
            st.write("Please upload both PDFs and enter a question.")

main()


