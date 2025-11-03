from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation"
)

chat_model = ChatHuggingFace(llm=llm)

st.header("Chatbot")
user_input = st.text_input("Enter user Prompt")


if st.button("Summerize"):
    result = chat_model.invoke(user_input)
    st.write(result.content)
