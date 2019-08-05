import torch
from pytorch_transformers import BertConfig, BertTokenizer, BertForTokenClassification

config = BertConfig.from_pretrained('bert-base-uncased', num_labels=7)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = BertForTokenClassification(config)
print(tokenizer.tokenize("Hello , my saturday is cute")) # ta ##co
input_ids = torch.tensor(tokenizer.encode("Hello , my taco is cute")).unsqueeze(0)  # Batch size 1
labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
outputs = model(input_ids, labels=labels)
loss, scores = outputs[:2]

print(scores.size())