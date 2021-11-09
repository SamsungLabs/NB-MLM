from pathlib import Path
import argparse
import numpy as np
import os

def read_data(data_path, mode):
    texts = []
    with open(data_path / f"{mode}.texts", "r") as fp:
        for line in fp:
            texts += [line]
    return texts


def prepare_mlm_all(origin_dir: str, data_dir: str, seed: int) -> None:
    np.random.seed(seed)

    data_path = Path(origin_dir)  
    print(f"Reading dataset from {str(data_path)}...", flush=True)

    train_texts = read_data(data_path, "train")
    dev_texts = read_data(data_path, "dev")
    test_texts = read_data(data_path, "test")

    train_size = len(train_texts)
    print(train_size)
    new_dev_size = int(0.05 * train_size)
    print(new_dev_size)
    if os.path.exists(data_path / "train_unlabeled.texts"):
        train_unlabeled_texts = read_data(data_path, "train_unlabeled")
        train_unlabeled_size = len(train_unlabeled_texts)

        dev_ids = np.random.choice(np.arange(train_unlabeled_size), size=new_dev_size, replace=False)
        new_dev_texts = [text for i, text in enumerate(train_unlabeled_texts) if i in dev_ids]
        train_unlabeled_texts = [text for i, text in enumerate(train_unlabeled_texts) if i not in dev_ids]

    else:
        dev_ids = np.random.choice(np.arange(train_size), size=new_dev_size, replace=False)
        new_dev_texts =  [text for i, text in enumerate(train_texts) if i in dev_ids]
        train_texts = [text for i, text in enumerate(train_texts) if i not in dev_ids]
    
    print(len(new_dev_texts))
    
    with open(Path(data_dir) / "train.texts", "w") as fp:
        fp.writelines(train_texts)
        fp.writelines(dev_texts)
        fp.writelines(test_texts)
        if os.path.exists(data_path / f"train_unlabeled.texts"):
            fp.writelines(train_unlabeled_texts)

    with open(Path(data_dir) / "dev.texts", "w") as fp:
        fp.writelines(new_dev_texts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin_dir", "-o", required=True, type=str)
    parser.add_argument("--data_dir", "-d", required=True, type=str)
    parser.add_argument("--seed", "-s", required=True, type=int)

    args = parser.parse_args()
    prepare_mlm_all(origin_dir=args.origin_dir, data_dir=args.data_dir, seed=args.seed)
