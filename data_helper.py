import json
import random
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer
import random
from sklearn.model_selection import  train_test_split,StratifiedKFold


def create_dataloaders(args, test_mode = False):
    val_ratio = args.val_ratio
    anns=list()
    with open(args.train_annotation,'r',encoding='utf8') as f:
        for line in f.readlines():
            ann =json.loads(line)
            anns.append(ann)
    random.shuffle(anns)
    val_anns = anns[:int(val_ratio*len(anns))]
    train_anns = anns[int(val_ratio*len(anns)):]
    # repeat <offline enhance>
    # train_anns = train_anns + train_anns
    val_dataset = MultiModalDataset(args, val_anns)
    train_dataset = MultiModalDataset(args, train_anns)
    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  sampler=train_sampler,
                                  drop_last=True,
                                  pin_memory=True,
                                  num_workers=args.num_workers,
                                  prefetch_factor=args.prefetch)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.val_batch_size,
                                sampler=val_sampler,
                                drop_last=False,
                                pin_memory=True,
                                num_workers=args.num_workers,
                                prefetch_factor=args.prefetch)
    return train_dataloader, val_dataloader

    
class MultiModalDataset(Dataset):
    def __init__(self,
                 args,
                 anns,
                 test_mode: bool = False,
                 idx= [] ):
        self.test_mode = test_mode
        if test_mode:
            self.tokenizer = BertTokenizer.from_pretrained(args.test_bert_dir)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
        self.anns=anns
    def __len__(self) -> int:
        return len(self.anns)
    def __getitem__(self, idx: int) -> dict:
        # id = self.anns[idx]['id']
        title = self.anns[idx]['title']
        assignee = self.anns[idx]['assignee']
        abstract = self.anns[idx]['abstract']
        # <online enhance here>
        # Step 2, load title tokens
        # text = title+assignee+abstract
        # text_inputs = self.tokenizer(title, max_length=512, padding='max_length', truncation=True)
        text_inputs = {}
        title_inputs = self.tokenizer(title, max_length=30, padding='max_length', truncation=True)
        assignee_inputs = self.tokenizer(assignee, max_length=15, padding='max_length', truncation=True)
        abstract_inputs = self.tokenizer(abstract, max_length= 450, padding='max_length', truncation=True)
        title_inputs['input_ids'][0] = 101
        assignee_inputs['input_ids'] = assignee_inputs['input_ids'][1:]
        abstract_inputs['input_ids'] = abstract_inputs['input_ids'][1:]
        assignee_inputs['attention_mask'] = assignee_inputs['attention_mask'][1:]
        abstract_inputs['attention_mask'] = abstract_inputs['attention_mask'][1:] 
        assignee_inputs['token_type_ids'] = assignee_inputs['token_type_ids'][1:]
        abstract_inputs['token_type_ids'] = abstract_inputs['token_type_ids'][1:] 
        for each in title_inputs:
            text_inputs[each] = title_inputs[each] + assignee_inputs[each] + abstract_inputs[each]
        text_inputs = {k: torch.LongTensor(v) for k,v in text_inputs.items()}
        text_mask = text_inputs['attention_mask']
        data = dict(
            text_inputs=text_inputs['input_ids'],
            text_mask=text_mask,
            text_type_ids = text_inputs['token_type_ids'],
        )
        # Step 4, load label if not test mode
        if (not self.test_mode):
            data['label'] = torch.LongTensor([self.anns[idx]['label_id']])
        return data