import json
from tqdm import tqdm
from nlpcda import Simbert

config = {
    'model_path': 'chinese_simbert_L-12_H-768_A-12',
    'CUDA_VISIBLE_DEVICES': '0,1',
    'max_len': 500,
    'seed': 1
}
simbert = Simbert(config=config)


def sim_bert_sentence(content):
    s = simbert.replace(sent=content, create_num=3)
    return s


anns = []
with open('data/new_train_aug_trans.json','r',encoding='utf8') as f:
    for line in f.readlines():
        ann =json.loads(line)
        anns.append(ann)

with open('new_train_aug_trans_bert.json', 'w', encoding='utf8') as f:
	for text in tqdm(anns):
		json.dump(text, f, ensure_ascii=False)
		f.write('\n')
		t = sim_bert_sentence(text['title'])
		ass = sim_bert_sentence(text['assignee'])
		abs = sim_bert_sentence(text['abstract'])
		for i in range(3):
			trans = {}
			trans['title'] = t[i][0]
			trans['assignee'] = ass[i][0]
			trans['abstract'] = abs[i][0]
			trans['label_id'] = text['label_id']
			json.dump(trans, f, ensure_ascii=False)
			f.write('\n')
