import argparse
from pathlib import Path
import typing as tp

import torch
import custom
from tqdm import tqdm
from fairseq.models.roberta import RobertaModel
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import os
import numpy as np

MAX_SEQLEN = 512
BATCH_SIZE = 128


def create_model(model_dir: str, data_path: str) -> RobertaModel:
    model = RobertaModel.from_pretrained(model_dir,
                                         checkpoint_file="checkpoint_best.pt",
                                         data_name_or_path=data_path)
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    model.to(device)
    model.eval()
    return model


def predict_batch(model: RobertaModel, batch: tp.List[str]) -> tp.List[int]:
    tokens_batch = torch.zeros(len(batch), MAX_SEQLEN, device=model.device)
    tokens_batch += model.task.dictionary.pad_index
    max_len = -1
    for i, text in enumerate(batch):
        tokens = model.encode(text)[:MAX_SEQLEN]
        cur_len = len(tokens)
        max_len = max(max_len, cur_len)
        tokens_batch[i, :cur_len] = tokens
    tokens_batch = tokens_batch[:, :max_len].long()
    with torch.no_grad():
        logits = model.predict("sentence_classification_head", tokens_batch)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        top_ids = probs.argmax(axis=-1)
    top_ids += model.task.label_dictionary.nspecial
    labels = model.task.label_dictionary.string(top_ids.tolist())
    return [label for label in labels.split()], probs


def evaluate_dataset(model_dir: str,
                     data_path: str,
                     mode: str,
                     batch_size: int = BATCH_SIZE,
                     f_score_avg: str = "macro",
                     save_dir: str = None,
                     rewrite_if_exists: bool = False
                     ):
    # Read test dataset
    fpreds, fmetrics = Path(save_dir) / f"predicted_{mode}.labels", Path(save_dir) / f"predicted_{mode}.metrics"
    if not rewrite_if_exists and fpreds.exists() and fmetrics.exists():
        print(f"{fpreds} and {fmetrics} exist: skipping checkpoint {model_dir}") 
        return

    texts, labels = [], []
    actual_data_path = Path(data_path[:-4])  # exclude '-bin'
    print(f"Reading dataset from {str(actual_data_path)}...", flush=True)

    with open(actual_data_path / f"{mode}.texts", "r") as fp:
        for line in fp:
            texts.append(line.strip())

    with open(actual_data_path / f"{mode}.labels", "r") as fp:
        for line in fp:
            labels.append(line.strip())
    print("Done.", flush=True)
    print(f"Total number of samples: {len(texts)}", flush=True)

    # Create Roberta model
    print(f"Loading model from {model_dir}", flush=True)
    model = create_model(model_dir, data_path)
    print("Done.", flush=True)

    predicted, probs = [], []
    print("Predicting...", flush=True)
    for sample_idx in tqdm(range(0, len(texts), batch_size)):
        prediction, batch_probs = predict_batch(model, texts[sample_idx: sample_idx + batch_size])
        predicted.extend(prediction)
        probs.append(batch_probs)

    print("Computing metrics") 
    metrics = {}
    metrics['accuracy'] = round(accuracy_score(labels, predicted) * 100, 4)
    metrics['err'] = 100 - metrics['accuracy']
    for f_score_avg in ['micro','macro']:
        metrics[f'f_{f_score_avg}'] = round(f1_score(labels, predicted, average=f_score_avg) * 100, 4)

    probs = np.vstack(probs)
    label_ids = [model.task.label_dictionary.index(l)-model.task.label_dictionary.nspecial for l in labels]
#    if probs.shape[-1]==2:
#        metrics['roc_auc_bin'] = roc_auc_score(label_ids, probs[:,-1])
#    else:
#        metrics['roc_auc'] = roc_auc_score(label_ids, probs)
    print(metrics)
    if save_dir:
        with open(fpreds, "w") as fp:
            fp.writelines([str(label) + "\n" for label in predicted])
        with open(fmetrics, "w") as fp:
            print(metrics, file=fp)


if __name__ == "__main__":
    print("Running evaluation...", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default="")
    parser.add_argument("--f-score-avg", type=str, default="macro")
    args = parser.parse_args()
    print(args, flush=True)
    evaluate_dataset(model_dir=args.model_dir,
                     data_path=args.data_path,
                     mode=args.mode,
                     f_score_avg=args.f_score_avg,
                     save_dir=args.save_dir)
