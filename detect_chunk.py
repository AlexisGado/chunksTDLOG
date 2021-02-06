from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction import DictVectorizer
import json
import matplotlib.pyplot as plt
import numpy as np
import random

# read the clean data
with open('data/chunks-cleaned.json', encoding="utf8") as json_data:
    chunks_dic = json.load(json_data)
with open('data/data-anonymized-cleaned.json', encoding="utf8") as json_data:
    messages = json.load(json_data)

# select all the chunks
chunks = []
for chunk in chunks_dic["chunks"]:
    for offset in chunk["selectionOffset"]:
        chunks.append(
            {"content": offset["content"], "length": offset["length"], "start": offset["offset_start"]})

# create non-chunks
all_candidates = []
esp = " "
for message in messages:
    if message["content"]:
        liste_mots = message["content"].split()
        n = len(liste_mots)
        for i in range(2):
            longueur = random.randint(0, n-1)
            start = random.randint(0, n-longueur-1)
            content = esp.join(liste_mots[start:start+longueur])
            debut = esp.join(liste_mots[0:start])
            all_candidates.append(
                {"content": content, "length": len(content), "start": len(debut)})


X = chunks + all_candidates
Y = np.concatenate(([1]*len(chunks), [0]*len(all_candidates)))
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42)

vec = DictVectorizer()
Xtrain = vec.fit_transform(X_train)
Xtest = vec.transform(X_test)


# GRADIENT BOOSTING CLASSIFIER
gbc = GradientBoostingClassifier(learning_rate=0.01, n_estimators=250, max_depth=50, min_samples_leaf=3,
                                 min_samples_split=100, subsample=0.9, random_state=42)

# to create sample weight array
gbc.fit(Xtrain, Y_train)

Y_pred = gbc.predict(Xtest)

print('accuracy %s' % accuracy_score(Y_pred, Y_test))

# test to study the model

# for i in range(len(Y_pred)):
#     if Y_pred[i] == 1 and Y_test[i] == 1:
#         print("this is chunk and the model recognize it : ", X_test[i])
#     if Y_test[i] == 1 and Y_pred[i] == 0:
#         print("the model does not recognize the chunk : ", X_test[i])
#     if Y_pred[i] == 1 and Y_test[i] == 0:
#         print(" this was not a chunk but the model thinks that : ", X_test[i])

message = messages[0]

esp = " "


def test(message):
    all_candidates2 = []
    content = message["content"]
    content_split = content.split(" ")
    for i in range(len(content_split)-1):
        for j in range(i+1, len(content_split)):
            candidate = esp.join(content_split[i:j])
            debut = esp.join(content_split[0:i])
            all_candidates2.append({"content": candidate, "length": len(
                candidate), "start": len(debut)})
    all_candidates2 = np.array(all_candidates2)
    Xtest2 = vec.transform(all_candidates2)
    Y_pred2 = gbc.predict(Xtest2)

    # displays all the chunks
    # for i in range(len(Y_pred2)):
    #     if Y_pred2[i]:
    #         print(all_candidates2[i])
