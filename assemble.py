from transformers import BertTokenizer, BertModel

# Load the tokenizer and model from the local directory
tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')
model = BertModel.from_pretrained('./bert-base-uncased')

# Example text
text = "Using BERT model offline is straightforward."

# Tokenize the input text
inputs = tokenizer(text, return_tensors='pt')

# Get the embeddings from BERT
outputs = model(**inputs)

# Extract the embeddings
last_hidden_states = outputs.last_hidden_state
print(last_hidden_states)
