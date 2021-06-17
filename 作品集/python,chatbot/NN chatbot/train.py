import numpy as np
import random
import json
import nltk
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk.stem.porter import PorterStemmer

from model import Net


with open('chatbox.json', 'r') as f:
    intents = json.load(f)
    
      
def BOW(tokenized_sentence, words):
    """
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # stem 
    sentence_words = [PorterStemmer().stem(word.lower()) for word in tokenized_sentence]
    # 全0**
    bag = np.zeros(len(words), dtype=np.float32)
    #bag = [0]*len(words)dtype=torch.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag

all_words = []
tags = []
documents = []
ignore = ['?', '.', '!', ',']

X_train = []
y_train = []
# loop through 
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = nltk.word_tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to documents
        documents.append((w, tag))
print("allwo:",all_words)
print("doc:",documents)
# stem and lower each word

all_words = [PorterStemmer().stem(w.lower()) for w in all_words if w not in ignore]
# remove duplicates and sort
all_words = sorted(list(set(all_words)))
tags = sorted(list(set(tags)))

# print(len(documents), "patterns")
# print(len(tags), "tags:", tags)
# print(len(all_words), "unique stemmed words:", all_words)

# create training data
for (pattern_sentence, tag) in documents:
    # X: bag of words 
    bag = BOW(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters 
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
n_input = len(X_train[0])
n_hidden = 8
n_output = len(tags)
print(n_input, n_output)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True)



model = Net(n_input, n_hidden, n_output)

# Loss and optimizer
CEloss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        labels = labels.to(dtype=torch.long)
        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = CEloss(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad() #清空梯度
        loss.backward() #反向传播
        optimizer.step() #更新模型参数
    if (epoch+1)%100==0:
        print (f'Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')




data = {
"model_state": model.state_dict(),
"n_input": n_input,
"n_hidden": n_hidden,
"n_output": n_output,
"all_words": all_words,
"tags": tags
}


torch.save(data, "data.pth")

print('training complete')
