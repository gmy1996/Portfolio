from tkinter import *
import csv
import nltk
from matplotlib import pyplot as plt
import numpy as np

window = Tk()
window.title("First Window")
window.geometry("600x150")
selected = StringVar()
lbl = Label(window, text="Which one do you prefer?")
lbl.grid(column=1, row=1)

rad1 = Radiobutton(window, text="Random", value="random",variable=selected)
rad2 = Radiobutton(window, text="Fixed", value="fixed",variable=selected)
rad3 = Radiobutton(window, text="Planned", value="planned",variable=selected)
rad1.grid(column=1, row=2)
rad2.grid(column=2, row=2)
rad3.grid(column=3, row=2)

lbl = Label(window, text="Why do you like it?")
lbl.grid(column=1, row=3)
txt = Entry(window, width=50)
txt.grid(column=3, row=4)

def clicked():
    result1=selected.get()
    result2=txt.get()
    tokens = nltk.word_tokenize(result2)
    print("tok:",tokens)
    tagged = nltk.pos_tag(tokens)
    print("tag:",tagged)
    properNouns=[word for (word,pos) in tagged if pos=="VBG" or pos=="JJ"]
    print(properNouns[0])

    data = {'reason':properNouns[0],'type':result1}
    with open(r'sheet.csv','a',newline='')as f:
        fieldnames = {'reason','type'}    # 表头
        writer = csv.DictWriter(f,fieldnames=fieldnames)
        #writer.writeheader()    
        writer.writerow(data)
        

btn = Button(window, text="Submit", command=clicked)
btn.grid(column=3, row=5)
window.mainloop()
pie=[]
bar=[]
with open('sheet.csv')as f:
    df = csv.reader(f)
    headers=next(df)

    for row in df:
        pie.append(row[0])
        #print(pie)
        bar.append(row[1])
    
        
num_random=str(pie).count("random")
num_planned=str(pie).count("planned")
num_fixed=str(pie).count("fixed")
t=[num_random,num_fixed,num_planned]
labels=["random","fixed","planned"]
plt.pie(t ,labels=labels,autopct='%1.1f%%',shadow=False,startangle=150)
plt.title("the pie chart of peoples' preference for three chatbots")
plt.show() 

