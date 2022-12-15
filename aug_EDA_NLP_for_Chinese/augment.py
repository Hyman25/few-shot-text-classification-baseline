import json
from tqdm import tqdm
from eda import *

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--num_aug", type=int, default=4, help="每条原始语句增强的语句数")
ap.add_argument("--alpha", type=float, default=0.3, help="每条语句中将会被改变的单词数占比")
args = ap.parse_args()

#每条原始语句增强的语句数
num_aug = 9 #default
if args.num_aug:
    num_aug = args.num_aug

#每条语句中将会被改变的单词数占比
alpha = 0.1 #default
if args.alpha:
    alpha = args.alpha


def get_eda_aug(content):
    aug_sentences = eda(content, alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha, p_rd=alpha, num_aug=num_aug)
    return aug_sentences

def gen_eda():
    ## 需要增强的文件
    sentences = []
    with open('new_train_aug_trans.json','r',encoding='utf8') as f:
        for line in f.readlines():
            ann =json.loads(line)
            sentences.append(ann)
    with open('new_train_aug_trans_eda.json', 'w', encoding='utf8') as f:
        for text in tqdm(sentences, ncols=120):
            json.dump(text, f, ensure_ascii=False)
            f.write('\n')
            t = get_eda_aug(text['title'])
            ass = get_eda_aug(text['assignee'])
            abs = get_eda_aug(text['abstract'])
            for i in range(num_aug):
                trans = {}
                trans['title'] = t[i]
                trans['assignee'] = ass[i]
                trans['abstract'] = abs[i]
                trans['label_id'] = text['label_id']
                json.dump(trans, f, ensure_ascii=False)
                f.write('\n')

if __name__ == "__main__":
    gen_eda()
