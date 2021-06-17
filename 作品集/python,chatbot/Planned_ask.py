from pprint import pprint
import random
import nltk



  
knowledge = { ("person1", "name", "?"), \
                 ("person1", "age", "?"),\
                 ("person1", "gender ", "?"),\
                
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
            question = "assistant: "+"What is your "+fact+"? "
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
            question = "assistant: "+"What is the brand of your shirt? "
            helpRequest = input(question)
            if helpRequest=="Bye":
                active=False
                continue
            listWords = nltk.word_tokenize(helpRequest)
            tagged = nltk.pos_tag(listWords)
            #print("how:",tagged)
            verbs=[word for (word,pos) in tagged if str.startswith(pos,"JJ") or str.startswith(pos,"NN")]
            knowledge.add(("person1","is wearing ",str(verbs[0]))) 
               
            question="assistant: "+"which brand do you like best ?"
            helpRequest = input(question)
            if helpRequest=="Bye":
                active=False
                continue
            listWords = nltk.word_tokenize(helpRequest)
            tagged = nltk.pos_tag(listWords)
            #print("how:",tagged)
            verbs1=[word for (word,pos) in tagged if str.startswith(pos,"JJ") or str.startswith(pos,"NN")]
            knowledge.add(("person1","favourite brand is",str(verbs1[0])))
               
            question="assistant: "+"The reason why do you like "+str(verbs1[0])+" ?"
            helpRequest = input(question)
            if helpRequest=="Bye":
                active=False
                continue
            listWords = nltk.word_tokenize(helpRequest)
            tagged = nltk.pos_tag(listWords)
            #print("how:",tagged)
            verbs2=[word for (word,pos) in tagged if str.startswith(pos,"JJ") or str.startswith(pos,"NN")]
            if any(item in {"price","comfortable","appearance","suitable"}for item in verbs2):
                question="assistant: "+"Is the "+str(verbs2[0])+" the first thing you consider when you buy clothes?"
                helpRequest = input(question)
                if helpRequest=="Bye":
                    active=False
                    continue
                listWords = nltk.word_tokenize(helpRequest)
                tagged = nltk.pos_tag(listWords)
                #print("how:",tagged)
                verbs3=[word for (word,pos) in tagged if str.startswith(pos,"UH") or str.startswith(pos,"DT")]
                if any(item in {"yes"} for item in verbs3):
                    knowledge.add(("person1","the reason for shopping",str(verbs3[0])))
                else:
                    print("assistant: OK, Thank you, bye")
                    active=False
            else:
                knowledge.add(("person1","the reason for shopping",str(verbs2[0])))
            print("knowledge:",knowledge - unknowns)
    
#if any(item in {"Nike","Adidas","Fila"} for item in verbs):


    
