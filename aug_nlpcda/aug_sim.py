import json
from tqdm import tqdm
from nlpcda import Similarword

def similar_sentence(content):
    smw = Similarword(create_num=2, change_rate=0.3)
    rs1 = smw.replace(content)
    return rs1[-1]

anns = []
with open('/raid/hh/AI/data/new_train_aug.json','r',encoding='utf8') as f:
        for line in f.readlines():
            ann =json.loads(line)
            anns.append(ann)

with open('new_train_aug_sim.json', 'w', encoding='utf8') as f:
    for text in tqdm(anns):
        trans = {}
        trans['title'] = similar_sentence(text['title'])
        trans['assignee'] = similar_sentence(text['assignee'])
        trans['abstract'] = similar_sentence(text['abstract'])
        trans['label_id'] = text['label_id']

        json.dump(text, f, ensure_ascii=False)
        f.write('\n')
        json.dump(trans, f, ensure_ascii=False)
        f.write('\n')