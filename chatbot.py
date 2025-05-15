import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
import streamlit as st
from groq import Groq  # Import the Groq client

class ChatbotManager:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en",
        device: str = "cpu",
        encode_kwargs: dict = {"normalize_embeddings": True},
        llm_model: str = "groq-api-model",  # Set the model to Groq API model
        llm_temperature: float = 0.7,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "vector_db",
        groq_api_key: str = "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"  # Groq API key
    ):
        self.model_name = model_name
        self.device = device
        self.encode_kwargs = encode_kwargs
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.groq_api_key = groq_api_key  # Store Groq API key

        # Initialize Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs=self.encode_kwargs,
        )

        # Define the prompt template
        self.prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer. Answer must be detailed and well explained.
Helpful answer:
"""

        # Initialize Qdrant client
        self.client = QdrantClient(
            url=self.qdrant_url, prefer_grpc=False
        )

        if self.collection_name not in [c.name for c in self.client.get_collections().collections]:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config={"size": 384, "distance": "Cosine"}
            )

        # Initialize the Qdrant vector store
        self.db = Qdrant(
            client=self.client,
            embeddings=self.embeddings,
            collection_name=self.collection_name
        )

        # Initialize the prompt
        self.prompt = ChatPromptTemplate.from_template(self.prompt_template)

        # Initialize the retriever
        self.retriever = self.db.as_retriever(search_kwargs={"k": 1})

        # Define chain type kwargs
        self.chain_type_kwargs = {"prompt": self.prompt}

        # Initialize the Groq client
        self.groq_client = Groq(api_key=self.groq_api_key)

    def get_groq_response(self, prompt: str) -> str:
        """
        Calls the Groq API to get a response for the given prompt using the Groq client.

        Args:
            prompt (str): The input text to generate a response for.

        Returns:
            str: The generated response from Groq API.
        """
        try:
            # Use the Groq client to generate a response
            completion = self.groq_client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",  # You can adjust the model name as needed
                messages=[{"role": "user", "content": prompt}],
                temperature=self.llm_temperature,
                max_completion_tokens=1024,
                top_p=1,
                stream=True,  # Set to True for streaming response
            )

            # Initialize an empty string to collect the response
            full_response = ""

            # Iterate over the streaming response (chunks)
            for chunk in completion:
                # Extract the content from each chunk and append it to full_response
                full_response += chunk.choices[0].delta.content or ""
                # You can print the content of each chunk if needed
                # print(chunk.choices[0].delta.content or "", end="")

            return full_response.strip()

        except Exception as e:
            st.error(f"⚠️ An error occurred with the Groq API: {e}")
            return "⚠️ Sorry, I couldn't process your request at the moment."

    def get_response(self, query: str) -> str:
        """
        Processes the user's query and returns the chatbot's response.

        Args:
            query (str): The user's input question.

        Returns:
            str: The chatbot's response.
        """
        try:
            # Get context from retriever
            context = self.retriever.get_relevant_documents(query)  # Fixed line here
            # Format the prompt
            formatted_prompt = self.prompt.format(context=context, question=query)

            # Get response from Groq API
            response = self.get_groq_response(formatted_prompt)
            return response  # 'response' is now a string containing only the 'result'
        except Exception as e:
            st.error(f"⚠️ An error occurred while processing your request: {e}")
            return "⚠️ Sorry, I couldn't process your request at the moment."
