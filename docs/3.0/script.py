import pandas as pd
import requests
import time

# --- 1. CONFIGURATION ---
EXCEL_FILE = "Evaluation.xlsx"
STREAMLIT_URL = "https://babaralishah-rag-llm-app.hf.space"

# --- 2. DEFINE YOUR ABLATION MATRIX MATRIX ---
# Adjust these values to run different experimental combinations!
TEST_CONFIG = {
    "eval_mode": "true",
    "chunks": "4",                      # Slider value
    "rerank": "true",                    # "true" or "false" string
    "hybrid": "false",                   # "true" or "false" string
    "ablation": "V1",                    # Your dropdown configuration (V1, V2, etc.)
    "rewriting_strategy": "none"         # Your query rewriting choice
}

# --- 3. LOAD EXCEL BENCHMARK ---
df = pd.read_excel(EXCEL_FILE)
successful_retrievals = 0
total_questions = len(df)

print(f"🚀 Launching Benchmark Matrix...")
print(f"⚙️ Params -> Phase: {TEST_CONFIG['ablation']} | Rerank: {TEST_CONFIG['rerank']} | Chunks: {TEST_CONFIG['chunks']}\n")

# --- 4. BENCHMARK EXECUTION LOOP ---
for index, row in df.iterrows():
    question = row['question']
    
    # Process the ground-truth target keys from Excel (assumes comma-separated)
    expected_keys_raw = str(row['all_relevant_sentence_keys'])
    expected_keys = [k.strip() for k in expected_keys_raw.split(',') if k.strip()]
    
    # Inject the specific question into the parameter payload package
    payload = TEST_CONFIG.copy()
    payload['query'] = question
    
    try:
        # Streamlit reads these values directly out of the URL query string
        response = requests.get(STREAMLIT_URL, params=payload)
        # --- ADDED DEBUG CODE ---
        print(f"Row {index+1} Debug - Status Code: {response.status_code}")
        if "application/json" not in response.headers.get("Content-Type", ""):
            print("⚠️ Server did NOT return JSON! Here is the beginning of what it returned:")
            print(response.text[:500]) # Prints the first 500 characters of the webpage/error
            break # Stop after row 1 to inspect the error
        # ------------------------
        
        response_data = response.json()
        
        if response.status_code == 200:
            response_data = response.json()
            retrieved_keys = response_data.get('retrieved_context_keys', [])
            
            # Check if all ground-truth keys were found in the database results
            all_found = all(key in retrieved_keys for key in expected_keys)
            
            if all_found:
                successful_retrievals += 1
                print(f"Row {index+1}: ✅ Match! All keys caught.")
            else:
                print(f"Row {index+1}: ❌ Miss. Wanted {expected_keys}, got {retrieved_keys}")
        else:
            print(f"Row {index+1}: ⚠️ HTTP Error {response.status_code} returned from server.")
            
    except Exception as e:
        print(f"Row {index+1}: ⚠️ Failed to evaluate row: {e}")
        
    # Micro delay to prevent overloading your Hugging Face Space CPU core
    time.sleep(0.1)

# --- 5. SCORING METRICS ---
retrieval_score = (successful_retrievals / total_questions) * 100
print("\n" + "="*45)
print(f"📊 RETRIEVAL ACCURACY SCORE: {retrieval_score:.2f}%")
print(f"🎯 Successfully hit {successful_retrievals} out of {total_questions} scenarios.")
print("="*45)
