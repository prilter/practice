''' SCRIPT FOR CHECKING DATASET '''

import json

def check_unique(fn):
    with open(fn, 'r', encoding='utf-8') as f:
        data = json.load(f)

    titles = []
    for item in data:
        if item['title'] in titles:
            return False
        titles.append(item['title'])

    return True

def getdatalen(fn):
    with open(fn, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return len(data)

fn = "json/data5000.json"

if check_unique(fn):
    print(f"Ok!  Len of uniqies: {getdatalen(fn)}")
else:
    print(f"Bad! All Len:        {getdatalen(fn)}")
