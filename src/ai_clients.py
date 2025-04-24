import os
import google.generativeai as genai
from groq import Groq
from openai import OpenAI
import ollama
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from .env file
load_dotenv()

def get_api_key(api_type: str) -> Optional[str]:
    """Gets the API key for the specified service from environment variables."""
    if api_type == "gemini":
        return os.getenv("GEMINI_API_KEY")
    elif api_type == "groq":
        return os.getenv("GROQ_API_KEY")
    elif api_type == "openai":
        return os.getenv("OPENAI_API_KEY")
    elif api_type == "ollama":
        # Ollama typically doesn't require an API key in the same way
        # It might rely on the service running locally or a specific base URL
        # Return a placeholder or handle based on how Ollama is set up
        return "ollama_configured" # Placeholder, adjust as needed
    else:
        return None

def get_gemini_client(api_key: str):
    """Configures and returns the Gemini client."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro') # Or specify a different model
        return model
    except Exception as e:
        print(f"Error configuring Gemini client: {e}")
        return None

def get_groq_client(api_key: str):
    """Configures and returns the Groq client."""
    try:
        client = Groq(api_key=api_key)
        return client
    except Exception as e:
        print(f"Error configuring Groq client: {e}")
        return None

def get_openai_client(api_key: str):
    """Configures and returns the OpenAI client."""
    try:
        client = OpenAI(api_key=api_key)
        return client
    except Exception as e:
        print(f"Error configuring OpenAI client: {e}")
        return None

def get_ollama_client(): # Ollama client might not need an API key
    """Configures and returns the Ollama client."""
    try:
        # The ollama library typically connects to a running Ollama service.
        # Configuration might involve setting OLLAMA_HOST environment variable
        # or specifying host in client initialization if needed.
        client = ollama.Client() # Add host='http://...' if needed
        # Verify connection by listing models, for example
        client.list()
        return client
    except Exception as e:
        print(f"Error configuring Ollama client: {e}. Ensure Ollama service is running.")
        return None

def extract_with_gemini(client, content: str, prompt: str) -> Optional[str]:
    """Uses the Gemini client to extract structured data from content based on a prompt."""
    full_prompt = f"{prompt}\n\nHere is the content:\n\n{content}"
    try:
        response = client.generate_content(full_prompt)
        return response.text
    except Exception as e:
        print(f"Error during Gemini extraction: {e}")
        return None

def extract_with_groq(client, content: str, prompt: str, model: str = "llama3-8b-8192") -> Optional[str]:
    """Uses the Groq client to extract structured data from content based on a prompt."""
    full_prompt = f"{prompt}\n\nHere is the content:\n\n{content}"
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": full_prompt,
                }
            ],
            model=model,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error during Groq extraction: {e}")
        return None

def extract_with_openai(client, content: str, prompt: str, model: str = "gpt-3.5-turbo") -> Optional[str]:
    """Uses the OpenAI client to extract structured data from content based on a prompt."""
    full_prompt = f"{prompt}\n\nHere is the content:\n\n{content}"
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant designed to extract information from text."
                },
                {
                    "role": "user",
                    "content": full_prompt,
                }
            ],
            model=model,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error during OpenAI extraction: {e}")
        return None

def extract_with_ollama(client, content: str, prompt: str, model: str = "llama3") -> Optional[str]:
    """Uses the Ollama client to extract structured data from content based on a prompt."""
    full_prompt = f"{prompt}\n\nHere is the content:\n\n{content}"
    try:
        response = client.chat(
            model=model,
            messages=[
                {
                    'role': 'user',
                    'content': full_prompt,
                },
            ]
        )
        return response['message']['content']
    except Exception as e:
        print(f"Error during Ollama extraction: {e}")
        return None