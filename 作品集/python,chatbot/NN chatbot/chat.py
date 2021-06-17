import random
import json
import numpy as np
import torch
from nltk.stem.porter import PorterStemmer
from model import Net
import nltk



with open('chatbox.json', 'r') as f:
    intents = json.load(f)
    
def BOW(tokenized_sentence, words):
    # stem each word
    sentence_words = [PorterStemmer().stem(word.lower()) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag

data = torch.load("data.pth")

n_input = data["n_input"]
n_hidden = data["n_hidden"]
n_output = data["n_output"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = Net(n_input, n_hidden, n_output)
model.load_state_dict(model_state)
model.eval()


print("assistant: Welcome! (type 'quit' to exit)")
while True:
    
    sentence = input("You: ")
    if sentence == "quit":
        break

    sentence = nltk.word_tokenize(sentence)
    X = BOW(sentence, all_words)
    #print(X.shape)
    X = X.reshape(1, X.shape[0])#重新排列变成1行n列
    X = torch.from_numpy(X)#得到张量
    
    
    output = model(X)
    #print(output)
    #计算准确率第一个tensor的value是不需要的，拿到第二个索引位置的值,索引每行最大值
    _, indices = torch.max(output, dim=1)
    #print("indices:",t)
    #取出单元素张量的元素值并返回该值，保持原元素类型不变
    tag = tags[indices.item()]
    #print(tag)
    #对张量使用softmax将张量缩放到(0,1)
    probs = torch.softmax(output, dim=1)
    #print("pbs:",probs)
    prob = probs[0][indices.item()]
    #print("probability:",prob.item())
    if prob.item() > 0.9:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"assistant: {random.choice(intent['responses'])}")
    else:
        print("assistant: sorry, please change other ways to answer.")