import json
import torch
from torch.utils.data import SequentialSampler, DataLoader
import os
from util import evaluate
from config import parse_args
from model import clsModel
from tqdm import tqdm 
from data_helper import MultiModalDataset
from util import *
# os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'


def validate(model, val_dataloader):
    model.eval()
    predictions = []
    labels = []
    losses = []
    with torch.no_grad():
        for batch in val_dataloader:
            loss, _, pred_label_id, label = model(batch)
            loss = loss.mean()
            predictions.extend(pred_label_id.cpu().numpy())
            labels.extend(label.cpu().numpy())
            losses.append(loss.cpu().numpy())
    loss = sum(losses) / len(losses)
    results = evaluate(predictions, labels)
    model.train()
    return loss, results


def inference():
    args = parse_args()
    print(args.ckpt_file)
    print(args.test_batch_size)
    anns=list()
    with open(args.test_annotation,'r',encoding='utf8') as f:
        for line in f.readlines():
            ann =json.loads(line)
            anns.append(ann)
    dataset = MultiModalDataset(args, anns)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=args.test_batch_size,
                            sampler=sampler,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=args.num_workers,
                            prefetch_factor=args.prefetch)
    # 2. load model
    model = clsModel(args)
    checkpoint = torch.load(args.ckpt_file, map_location='cpu')
    new_key = model.load_state_dict(checkpoint['model_state_dict'],strict=False)
    # model.half()
    if torch.cuda.is_available():
        model = torch.nn.parallel.DataParallel(model.cuda())
    model.eval()
    loss, results = validate(model, dataloader)
    results = {k: round(v, 4) for k, v in results.items()}
    logging.info(results)


if __name__ == '__main__':
    setup_logging()
    inference()