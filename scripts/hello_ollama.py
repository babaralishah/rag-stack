import sys
import time
import threading
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"

SYSTEM_PROMPT = """You are a helpful assistant.
Follow the user's instructions carefully.
If you are unsure, say so briefly and ask a clarifying question.
"""

def build_prompt(system_prompt: str, user_msg: str) -> str:
    return f"""System:
{system_prompt.strip()}

User:
{user_msg.strip()}

Assistant:"""

def ollama_generate(prompt: str, context=None, temperature: float = 0.1):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False, # False: Wait for the full answer. True: Send words one-by-one as they generate.
        "temperature": temperature, # 0.1 for RAG/Facts, 0.5 for Chat, 1.0 for Creative Writing
    }
    if context is not None:
        payload["context"] = context

    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    return data.get("response", ""), data.get("context", None)

def main():
    global SYSTEM_PROMPT

    print(f"Ollama CLI Chat — model: {MODEL}")
    print("Commands: /exit /quit /clear")
    print("System prompt commands: /sys | /sys set ... | /sys add ... | /sys clear\n")

    context = None
    temperature = 0.1

    while True:
        try:
            user_msg = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user_msg:
            continue

        # --- commands ---
        if user_msg.lower() in ("/exit", "/quit"):
            print("Bye!")
            break

        if user_msg.lower() == "/clear":
            context = None
            print("(context cleared)\n")
            continue

        if user_msg.lower() == "/sys":
            print("\n=== SYSTEM PROMPT ===")
            print(SYSTEM_PROMPT.strip() or "(empty)")
            print("=====================\n")
            continue

        if user_msg.lower().startswith("/sys clear"):
            SYSTEM_PROMPT = ""
            print("(system prompt cleared)\n")
            continue

        if user_msg.lower().startswith("/sys set "):
            SYSTEM_PROMPT = user_msg[len("/sys set "):]
            print("(system prompt replaced)\n")
            continue

        if user_msg.lower().startswith("/sys add "):
            addition = user_msg[len("/sys add "):]
            if SYSTEM_PROMPT and not SYSTEM_PROMPT.endswith("\n"):
                SYSTEM_PROMPT += "\n"
            SYSTEM_PROMPT += addition + "\n"
            print("(system prompt appended)\n")
            continue

        # --- build engineered prompt + user prompt ---
        prompt = build_prompt(SYSTEM_PROMPT, user_msg)

        stop_event = threading.Event()
        t = threading.Thread(target=spinner, args=(stop_event,))
        t.start()

        try:
            reply, context = ollama_generate(prompt=prompt, context=context, temperature=temperature)
        except requests.exceptions.ConnectionError:
            print("\nERROR: Cannot connect to Ollama at http://localhost:11434")
            print("Fix: Run `ollama serve` (or open the Ollama app).")
            continue
        except requests.exceptions.Timeout:
            print("\nERROR: Request timed out.")
            continue
        except requests.exceptions.HTTPError as e:
            print("\nERROR: HTTP error:", e)
            continue
        finally:
            stop_event.set()
            t.join()

        print("Assistant:", reply.strip(), "\n")

def spinner(stop_event: threading.Event, message="Assistant: thinking "):
    frames = ["|", "/", "-", "\\"]
    i = 0
    while not stop_event.is_set():
        sys.stdout.write("\r" + message + frames[i % len(frames)])
        sys.stdout.flush()
        i += 1
        time.sleep(0.1)
    sys.stdout.write("\r" + " " * (len(message) + 2) + "\r")
    sys.stdout.flush()


if __name__ == "__main__":
    main()


# import json
# import sys
# import requests

# OLLAMA_URL = "http://localhost:11434/api/generate"
# # MODEL = "llama3.1:latest"
# MODEL = "llama3.2:3b"

# def main():
#     # prompt = "Say hello in one sentence, and tell me the number 7 squared."
#     prompt = "Explain RAG (LLMs) in simple terms."
#     payload = {
#         "model": MODEL,
#         "prompt": prompt,
#         "stream": False, # False: Wait for the full answer. True: Send words one-by-one as they generate.
#         "temperature": 0.1, # 0.1 for RAG/Facts, 0.5 for Chat, 1.0 for Creative Writing
#     }

#     try:
#         r = requests.post(OLLAMA_URL, json=payload, timeout=60)
#         r.raise_for_status()
#     except requests.exceptions.ConnectionError:
#         print("ERROR: Cannot connect to Ollama at http://localhost:11434")
#         print("Fix: Make sure Ollama is installed and running. Try: `ollama serve` (or open the Ollama app).")
#         sys.exit(1)
#     except requests.exceptions.HTTPError as e:
#         print("ERROR: Ollama returned an HTTP error:", e)
#         print("Response:", r.text)
#         sys.exit(1)
#     except requests.exceptions.Timeout:
#         print("ERROR: Request timed out. The model may still be downloading or your machine is slow.")
#         print("Fix: Try again, or use a smaller model like llama3.2:3b.")
#         sys.exit(1)

#     data = r.json()
#     print("\n=== Ollama Response ===")
#     print(data.get("response", "").strip())

# if __name__ == "__main__":
#     main()