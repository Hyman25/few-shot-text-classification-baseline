import json
from tqdm import tqdm
from nlpcda import RandomDeleteChar

def similar_sentence(content):
    smw = RandomDeleteChar(create_num=2, change_rate=0.3)
    rs1 = smw.replace(content)
    return rs1[-1]

anns = []
with open('new_train_aug_trans.json','r',encoding='utf8') as f:
        for line in f.readlines():
            ann =json.loads(line)
            anns.append(ann)

with open('new_train_aug.json', 'w', encoding='utf8') as f:
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