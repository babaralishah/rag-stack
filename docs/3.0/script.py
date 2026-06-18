import pandas as pd
import requests
import json
import time

# --- 1. CONFIGURATION ---
EXCEL_FILE = "Evaluation.xlsx"
# Copilot started your background FastAPI server on port 8001:
LOCAL_API_URL = "http://127.0.0.1:8001/eval"

# Define the exact ablation configuration you want to test right now
TEST_CONFIG = {
    "chunks": 4,                        # Integer
    "rerank": True,                     # Boolean (True/False)
    "hybrid": False,                    # Boolean (True/False)
    "ablation": "V1",                    # String choice
    "rewriting_strategy": "none"         # String choice
}

# --- 2. LOAD EXCEL BENCHMARK ---
try:
    df = pd.read_excel(EXCEL_FILE)
except Exception as e:
    print(f"❌ Error reading {EXCEL_FILE}: {e}")
    exit()

successful_retrievals = 0
total_questions = len(df)

print(f"🚀 Launching Local API Benchmark Matrix...")
print(f"⚙️ Target Endpoint -> {LOCAL_API_URL}")
print(f"⚙️ Params -> Phase: {TEST_CONFIG['ablation']} | Rerank: {TEST_CONFIG['rerank']} | Chunks: {TEST_CONFIG['chunks']}\n")

# --- 3. BENCHMARK EXECUTION LOOP ---
for index, row in df.iterrows():
    question = row['question']
    
    # Process and clean the ground-truth target keys from Excel (handles brackets, quotes, and spaces)
    expected_keys_raw = str(row['all_relevant_sentence_keys'])
    for char in ['[', ']', '"', "'"]:
        expected_keys_raw = expected_keys_raw.replace(char, '')
    expected_keys = [k.strip() for k in expected_keys_raw.split(',') if k.strip()]
    
    # Construct the JSON payload package matching Copilot's FastAPI Pydantic schema
    payload = TEST_CONFIG.copy()
    payload['query'] = question
    
    try:
        # Copilot configured the FastAPI endpoint to listen for POST requests
        response = requests.post(LOCAL_API_URL, json=payload, headers={"Content-Type": "application/json"})
        
        if response.status_code == 200:
            response_data = response.json()
            retrieved_keys = response_data.get('retrieved_context_keys', [])
            
            # Clean up whatever formats come out of your vector db keys to match Excel strings
            retrieved_keys_clean = [str(k).replace('"', '').replace("'", "").strip() for k in retrieved_keys]
            
            # Check if all ground-truth keys were found in the database results
            all_found = all(key in retrieved_keys_clean for key in expected_keys)
            
            if all_found:
                successful_retrievals += 1
                print(f"Row {index+1}: ✅ Match! All keys caught.")
            else:
                print(f"Row {index+1}: ❌ Miss. Wanted {expected_keys}, got {retrieved_keys_clean}")
        else:
            print(f"Row {index+1}: ⚠️ HTTP Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"Row {index+1}: ⚠️ Failed to connect/evaluate row: {e}")
        
    # Micro pause to prevent slamming your local CPU threads
    time.sleep(0.05)

# --- 4. SCORING METRICS ---
retrieval_score = (successful_retrievals / total_questions) * 100
print("\n" + "="*45)
print(f"📊 RETRIEVAL ACCURACY SCORE: {retrieval_score:.2f}%")
print(f"🎯 Successfully hit {successful_retrievals} out of {total_questions} scenarios.")
print("="*45)
