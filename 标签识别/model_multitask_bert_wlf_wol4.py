import torch
from transformers import BertTokenizer, BertForMaskedLM

# Load pre-trained model and tokenizer
model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

# Sample input text
text = "我喜欢吃[MASK]。"

# Tokenize the text
tokens = tokenizer.tokenize(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text)))
inputs = tokenizer.encode(text, return_tensors="pt")

# Mask a token (in this example, the last token)
mask_position = -2
inputs[0][mask_position] = tokenizer.mask_token_id

# Get the model's predictions
with torch.no_grad():
    outputs = model(inputs)

# Get the predicted token for the masked position
predicted_token_id = torch.argmax(outputs.logits[0, mask_position]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_token_id])[0]

print("Masked Sentence:", text)
print("Predicted Token for [MASK]:", predicted_token)
