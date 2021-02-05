import json
import re


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


with open('data/data-anonymized.json', encoding="utf8") as json_data:
    messages = json.load(json_data)


for message in messages:
    for key in ["requester", "submitter"]:
        if message[key]["email"]:
            message[key]["email"] = message[key]["email"].lower().replace(" ", ".")
    for submessage in message["sub_messages"]:
        if submessage["submitter"]["email"]:
            submessage["submitter"]["email"] = submessage["submitter"]["email"].lower(
            ).replace(" ", ".")
    if message["content"]:
        message["content"] = cleanhtml(message["content"])
    for submessage in message["sub_messages"]:
        if submessage["content"]:
            submessage["content"] = cleanhtml(submessage["content"])


with open('data/data-anonymized-cleaned.json', 'w') as outfile:
    json.dump(messages, outfile)
