from transformers import T5ForConditionalGeneration, T5Tokenizer

# Yeh model specially 2-person chat/dialogue ke liye train hua hai
model_name = "mrm8488/t5-base-finetuned-samsum"

model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)

# Yeh automatically saari zaroori files (Dimaag + Translator) aapke VS Code wale folder mein daal dega
model.save_pretrained("./Saved_Summarizer_Model")
tokenizer.save_pretrained("./Saved_Summarizer_Model")

print("Success! Naya Chat Expert model aapke folder mein save ho gaya hai.")