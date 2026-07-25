from datasets import load_dataset
import pandas as pd

print("Streaming a tiny sample from HotpotQA...")

# 1. Stream the validation split to save local disk space entirely
dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", streaming=True)

# 2. Extract a small sample (100 items will be < 1MB on your disk)
SAMPLE_SIZE = 100
small_sample = list(dataset.take(SAMPLE_SIZE))

qa_list = []
unique_sources = set()

# 3. Process the mini dataset
for item in small_sample:
    qa_list.append({
        "ID": item["id"],
        "Question": item["question"],
        "Correct Answer": item["answer"],
        "Type": item["type"],
        "Level": item["level"]
    })
    
    # Extract the context documents
    context_dict = item["context"]
    for title, sentences in zip(context_dict["title"], context_dict["sentences"]):
        paragraph = "".join(sentences)
        full_doc = f"Title: {title}\n{paragraph}"
        unique_sources.add(full_doc)

# 4. Save the tiny QAs list to Excel
excel_filename = "hotpotqa_mini_qas.xlsx"
df_qas = pd.DataFrame(qa_list)
df_qas.to_excel(excel_filename, index=False)

# 5. Save the tiny knowledge base to a text file
txt_filename = "hotpotqa_mini_sources.txt"
with open(txt_filename, "w", encoding="utf-8") as f:
    for doc in unique_sources:
        f.write(doc + "\n\n" + "="*50 + "\n\n")

print(f"\nSuccess! Generated 2 tiny files:")
print(f"1. QA Excel Sheet: '{excel_filename}'")
print(f"2. Source Text Corpus: '{txt_filename}'")