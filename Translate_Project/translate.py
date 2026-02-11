from transformers import MarianMTModel, MarianTokenizer

# 1. Define model
model_name = "Helsinki-NLP/opus-mt-en-hi"

print("Initializing Model... (Connecting brain and dictionary)")

try:
    # 2. Load the brain (model) and the dictionary (tokenizer)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    print("\n--- AI English to Hindi Translator ---")
    src_text = input("Enter English sentence: ")

    # 3. Tokenize the text (convert words to numbers for the AI)
    tokenized_text = tokenizer(src_text, return_tensors="pt", padding=True)

    # 4. Generate the translation
    translated_tokens = model.generate(**tokenized_text)

    # 5. Decode back to words (convert numbers back to Hindi words)
    hindi_text = tokenizer.decode(translated_tokens[0], skip_special_code_tokens=True, skip_special_tokens=True)

    print("-" * 30)
    print(f"English: {src_text}")
    print(f"Hindi: {hindi_text}")
    print("-" * 30)

except Exception as e:
    print(f"\n[ERROR]: {e}")