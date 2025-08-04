import pandas as pd
from transformers import pipeline
from tqdm import tqdm

# Load your cleaned training file
with open("gpt2_coursegen_data_cleaned.txt", "r", encoding="utf-8") as f:
    examples = f.read().strip().split("\n\n")

# Extract summaries from each training example
entries = []
for ex in examples:
    if "<|endofprompt|>" not in ex:
        continue
    prompt, summary = ex.split("<|endofprompt|>", 1)
    entries.append((prompt.strip(), summary.strip()))

# Load the FLAN-T5 model
formalizer = pipeline("text2text-generation", model="google/flan-t5-large", device=0)  # set device=0 if GPU available

# Formalize summaries
formalized_data = []
for prompt, summary in tqdm(entries, desc="Formalizing summaries"):
    input_text = f"Rewrite the following in a formal, third-person academic tone:\n\n{summary}"
    try:
        result = formalizer(input_text, max_length=100, do_sample=False)[0]['generated_text']
    except Exception as e:
        result = "Error formalizing summary"
    formalized_data.append((prompt, result.strip()))

# Save formalized output
with open("gpt2_coursegen_data_formalized.txt", "w", encoding="utf-8") as f:
    for prompt, summary in formalized_data:
        f.write(f"{prompt}\n<|endofprompt|>{summary}\n\n")

print("✅ Formalized file saved as: gpt2_coursegen_data_formalized.txt")
