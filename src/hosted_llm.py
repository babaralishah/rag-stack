import os
from groq import Groq

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is required. Set it in Hugging Face Secrets or locally.")
    return Groq(api_key=api_key)

def generate_answer(prompt: str) -> str:
    try:
        client = get_groq_client()
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",   # or "llama-3.3-70b-versatile" if you upgrade HF hardware
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM Error: {str(e)}"

# import os
# from groq import Groq

# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# client = Groq(api_key=GROQ_API_KEY)

# def generate_answer(prompt: str) -> str:
#     try:
#         response = client.chat.completions.create(
#             model="llama-3.1-8b-instant",
#             messages=[
#                 {"role": "user", "content": prompt}
#             ],
#         )

#         return response.choices[0].message.content.strip()

#     except Exception as e:
#         return f"Error generating answer: {str(e)}"