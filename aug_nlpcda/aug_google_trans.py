import json
from tqdm import tqdm
import re
import html
from urllib import parse
import requests

GOOGLE_TRANSLATE_URL = 'http://translate.google.com/m?q=%s&tl=%s&sl=%s'

def google_translate(text, to_language="auto", text_language="auto"):

    text = parse.quote(text)
    url = GOOGLE_TRANSLATE_URL % (text,to_language,text_language)
    response = requests.get(url)
    data = response.text
    expr = r'(?s)class="(?:t0|result-container)">(.*?)<'
    result = re.findall(expr, data)
    if (len(result) == 0):
        return ""

    return html.unescape(result[0])


def translate_twice(content):
    en_s = google_translate(content, to_language="en", text_language="zh-CN")
    zh_s = google_translate(en_s, to_language="zh-CN", text_language="en")
    return zh_s


anns = []
with open('new_train.json','r',encoding='utf8') as f:
        for line in f.readlines():
            ann =json.loads(line)
            anns.append(ann)

with open('new_train_aug.json', 'w', encoding='utf8') as f:
    for text in tqdm(anns):
        trans = {}
        trans['title'] = translate_twice(text['title'])
        trans['assignee'] = translate_twice(text['assignee'])
        trans['abstract'] = translate_twice(text['abstract'])
        trans['label_id'] = text['label_id']

        json.dump(text, f, ensure_ascii=False)
        f.write('\n')
        json.dump(trans, f, ensure_ascii=False)
        f.write('\n')