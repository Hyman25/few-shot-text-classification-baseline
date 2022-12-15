import requests
import random
import hashlib
import json
from tqdm import tqdm

# 百度翻译方法
def baidu_translate(content, t_from='en', t_to='zh', appid='xxx', secretKey='xxx'):
    # print(content)
    if len(content) > 4891:
        return '输入请不要超过4891个字符！'
    salt = str(random.randint(0, 50))
    # 申请网站 http://api.fanyi.baidu.com/api/trans
    # 这里写你自己申请的
    appid = appid
    # 这里写你自己申请的
    secretKey = secretKey
    sign = appid + content + salt + secretKey
    sign = hashlib.md5(sign.encode(encoding='UTF-8')).hexdigest()
    head = {'q': f'{content}',
            'from': t_from,
            'to': t_to,
            'appid': f'{appid}',
            'salt': f'{salt}',
            'sign': f'{sign}'}
    j = requests.get('http://api.fanyi.baidu.com/api/trans/vip/translate', head)
    # print(j.json())
    res = j.json()['trans_result'][0]['dst']
    # print(res)
    return res


def translate_twice(content):
    en_s = baidu_translate(content=content, t_from='zh', t_to='en')
    zh_s = baidu_translate(content=en_s, t_from='en', t_to='zh')
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