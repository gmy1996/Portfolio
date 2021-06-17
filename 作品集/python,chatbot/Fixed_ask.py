from pprint import pprint
import random
import nltk

knowledge2 = { ("person1", "age", "?"),
             ("person1", "gender", "?"),
             ("person1", "name", "?"),
             ("person1", "shirt brand you are wearing", "?"), 
             ("person1", "favourite brand of clothes or shoes", "?"),
             ("person1", "shirt's price", "?")
             }

active = True
while active:
    unknowns = { (person,fact,value) for (person,fact,value) \
                 in knowledge2 if value=="?" }
    #print("UNKNOWN:")
    #pprint(unknowns)
    #print("KNOWN:")
    #pprint(knowledge2 - unknowns)
    if unknowns: #is non-empty
        for i in list(unknowns):
            #print("i:",i)
            person, fact, value = i
            question = "What is your "+fact+"? "
            knowledge2.remove( (person,fact,value) )
            reply = input(question)
            if reply=="bye":
                active = False
                continue
            tokens = nltk.word_tokenize(reply)
            tagged = nltk.pos_tag(tokens)
            properNouns = [ word for (word, pos) in tagged if pos=="NNP" ]
            #print("tagged",tagged)
            #print("ProperNouns:",properNouns)
            if not properNouns:
                properNouns = [ word for (word, pos) in tagged if pos=="NN" ]
                if not properNouns:
                    properNouns=[word for (word,pos) in tagged if pos=="CD"]
                knowledge2.add( (person, fact, properNouns[0]) )
            knowledge2.add( (person, fact, properNouns[0]) )
    else:
        print("Final knowledge base: ")
        pprint(knowledge2)
        active=False
  

    
    
