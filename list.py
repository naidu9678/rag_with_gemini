# list.py

import os
import google.generativeai as genai

def list_available_models():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Please set your GOOGLE_API_KEY environment variable.")

    genai.configure(api_key=api_key)

    print("\n✅ Available Gemini models:\n")
    for model in genai.list_models():
        print(f" • {model.name} → {model.supported_generation_methods}")

if __name__ == "__main__":
    list_available_models()
