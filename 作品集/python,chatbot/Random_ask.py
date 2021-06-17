from pprint import pprint
import random
import nltk
from tkinter import *


  
knowledge = { ("person1", "favorite brand", "?"), \
                 ("person1", "shirt brand you are wearing", "?"),\
                 ("person1", "name ", "?"),\
                 ("person1", "age ", "?"),\
                 ("person1", "gender ", "?"),\
                 ("person1", "shirt's price", "?"),\
                 }

active = True
while active:
        unknowns = { (person,fact,value) for (person,fact,value) \
                     in knowledge if value=="?" }
        #print("UNKNOWN:")
        #pprint(unknowns)
        #print("KNOWN:")
        #pprint(knowledge - unknowns)
        if unknowns: #is non-empty
            person, fact, value = random.choice(list(unknowns))
            question = "What is your "+fact+"? "
            reply=input(question)
            knowledge.remove( (person,fact,value) )
            if reply=="bye":
                active = False
                continue
            tokens = nltk.word_tokenize(reply)
            tagged = nltk.pos_tag(tokens)
            properNouns = [ word for (word, pos) in tagged if pos=="NNP" ]
            # print("tagged",tagged)
            # print("ProperNouns:",properNouns)
            if not properNouns:
                properNouns = [ word for (word, pos) in tagged if pos=="NN" ]
                if not properNouns:
                    properNouns=[word for (word,pos) in tagged if pos=="CD"]
                knowledge.add( (person, fact, properNouns[0]) )
            knowledge.add( (person, fact, properNouns[0]) )
        else:
            print("Final knowledge base: ")
            pprint(knowledge)
            active=False            
    
    



