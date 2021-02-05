import json
import re


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


with open('data/ponts-chunk.json', encoding="utf8") as json_data:
    chunks_dic = json.load(json_data)


for chunk in chunks_dic["chunks"]:
    for offset in chunk["selectionOffset"]:
        if offset["content"]:
            offset["content"] = cleanhtml(offset["content"])

with open('data/chunks-cleaned.json', 'w') as outfile:
    json.dump(chunks_dic, outfile)
