import matplotlib.pyplot as plt
import numpy as np
import random
import json
import os
import panda as pd


with open("data\chunks-cleaned.json", encoding="utf8") as json_data:
    chunks_dic_clean = json.load(json_data)
with open("data\ponts-chunk.json", encoding="utf8") as json_data:
    chunks_dic = json.load(json_data)

with open('data/data-anonymized-cleaned.json', encoding="utf8") as json_data:
    messages_clean = json.load(json_data)
with open('data/data-anonymized.json', encoding="utf8") as json_data:
    messages = json.load(json_data)


print(len(messages))
chunks = chunks_dic["chunks"]


longueurs = [len(chunk["selectionOffset"][i]["content"].split())
             for chunk in chunks for i in range(len(chunk["selectionOffset"]))]

# plt.hist(longueurs, range=(0, 50), bins=51)
pos = []
nb_zones = 10
for chunk in chunks:
    for message in messages:
        if (message["id"] == chunk["message"]["id"]):
            longueur = len(message["content"])
            for i in range(len(chunk["selectionOffset"])):
                for l in range(nb_zones):
                    if chunk["selectionOffset"][i]["offset_start"] < (l+1)*longueur/nb_zones and chunk["selectionOffset"][i]["offset_start"]+chunk["selectionOffset"][i]["length"] >= l*longueur/nb_zones:
                        pos.append(l+1)

plt.hist(pos, range=(1, nb_zones+1), bins=nb_zones)
plt.show()


#print(pd.get_dummies("bonjour comment ca va ? Bien et vous."))

# Features importants : longueur, position,
