import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class clsModel(nn.Module):
    def __init__(self, args):
        super(clsModel, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_dir, output_hidden_states=True)
        # config = BertConfig(output_hidden_states=True)
        # self.bert = BertModel(config=config)
        class_num = 32
        self.cls = nn.Linear(768*4, class_num)
        self.text_embedding = self.bert.embeddings
        self.text_cls = nn.Linear(768, class_num)
    def build_pre_input(self, data):
        text_inputs=data['text_inputs']
        text_mask=data['text_mask']
        textembedding = self.text_embedding(text_inputs.cuda(), data['text_type_ids'].cuda())
        return textembedding,text_mask
    def forward(self, data, inference=False,multi = False):
        inputs_embeds, mask = self.build_pre_input(data)
        bert_out = self.bert(attention_mask=mask, inputs_embeds=inputs_embeds)
        # last 4 mean pooling
        hidden_stats = bert_out.hidden_states[-4:]
        hidden_stats = [i.mean(dim=1) for i in hidden_stats]
        out = self.cls(torch.cat(hidden_stats,dim=1))
        if inference:
            if multi:
                return out
            else:
                return torch.argmax(out, dim=1)
        else:
            all_loss, all_acc, all_pre,label = self.cal_loss(out,data['label'].cuda())
            return all_loss, all_acc, all_pre, label
    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label
