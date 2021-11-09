from pathlib import Path
import argparse
import numpy as np


def strip_dict(d):
    return {k: str(v).replace("\n", " ").strip() for k, v in d.items()}


def process_file(name: str, dataset: str, path: str) -> None:

    with open(path, "r") as fp:
        lines = [
            strip_dict(eval(line, {"null": ""}))
            for line in fp
            if line.strip()
        ]
    
    if dataset == "ag":
        texts = []
        for line in lines:
            if len(line["headline"]) > 0:
                if (line["headline"][-1].isdigit() or 
                        line["headline"][-1].isalpha() or
                        line["headline"][-1] in [")", "'"]):
                        
                        line["headline"] = line["headline"] + '.'
                elif line["headline"][-1] not in ['!', '?']:
                    line["headline"] = line["headline"][:-1] + '.'
                texts += [line["headline"] + " " + line["text"] + "\n"]
            else:
                texts += [line["text"] + "\n"]
    else:
        texts = [line["text"] + "\n" for line in lines]
    
    labels = np.array([line["label"] for line in lines])
    
    save_dir = Path(path).parent
    dict_path = save_dir / "dict.txt"
    
    if name == 'train':
        label2idx = {label:str(i) for i, label in enumerate(np.unique(labels))}
        with dict_path.open("w") as fp:
            for label, idx in label2idx.items():
                fp.write(label + ' ' + idx + '\n')
    else:
        label2idx = {}
        with dict_path.open("r") as fp:
            for line in fp:
                label, idx = line.split()
                label2idx[label] = idx
    
    labels = [label2idx[label] + "\n" for label in labels]

    texts_path = save_dir / f"{name}.texts"
    labels_path = save_dir / f"{name}.labels"
    with texts_path.open("w") as tfp, labels_path.open("w") as lfp:
        for i in range(len(texts)):
            try:
                text, label = texts[i], labels[i]
                if len(text) > 1:
                    tfp.write(text)
                    lfp.write(label)
            except Exception as e:
                print(f"Filer row {i}: ", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", "-n", required=True, type=str)
    parser.add_argument("--dataset", "-d", required=True, type=str)
    parser.add_argument("--path", "-p", required=True, type=str)

    args = parser.parse_args()
    process_file(name=args.name, dataset=args.dataset, path=args.path)
