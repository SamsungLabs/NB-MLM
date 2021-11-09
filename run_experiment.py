import sys
import json
import re
import argparse
from pathlib import Path
from shutil import copyfile

from fairseq_cli.train import cli_main as train
from _jsonnet import evaluate_file


def to_path(dirname: str, filename: str):
    return str(Path(dirname) / filename)


DIR = Path(__file__).parent.resolve()
EXT_VARS = {
    "USER_DIR": str(DIR / "custom"),
    "FAIRSEQ_DIR": str(DIR / "fairseq-mlm-pretrain"),
}

CALLBACKS = {
    "to_path": (("dirname", "filename"), to_path),
}


def main(args: argparse.Namespace):
    dataset, full_task = args.config_path.split("/")
    config_path = str(DIR / "configs" / full_task) + ".jsonnet"
    task = "mlm" if "mlm" in full_task else "clf"

    hparams_config_dir = DIR / "configs" / dataset
    dataset_part = args.dataset_part
    if dataset_part:
        hparams_config_path = hparams_config_dir / f"{task}-{dataset_part}-hparams.jsonnet"
        if not hparams_config_path.exists():
            hparams_config_path = hparams_config_dir / f"{task}-hparams.jsonnet"
    else:
        hparams_config_path = hparams_config_dir / f"{task}-hparams.jsonnet"
    hparams_config_path = str(hparams_config_path)

    experiment_dir = DIR / f"{dataset}_experiments"
    (experiment_dir / "RUNS").mkdir(exist_ok=True)

    full_task = full_task.replace("-", "_")
    run_dir = experiment_dir / "RUNS"
    if dataset_part:
        run_dir /= f"{dataset}_{dataset_part}_{full_task}"
    else:
        run_dir /= f"{dataset}_{full_task}"
    run_dir = str(run_dir)

    for arg in sys.argv[2:]:
        parts = arg.split("=")
        argname, argvalue = parts[0], True
        if len(parts) == 2:
            argvalue = parts[1]
        argname = argname[2:].replace("-", "_")
        if argname == "dataset_part":
            continue
        if not isinstance(argvalue, bool):
            EXT_VARS[argname.upper()] = argvalue

        run_dir += f"_{argname}_{argvalue}"

    data_dir = experiment_dir / "DATA"
    if dataset_part:
        data_dir /= f"{dataset}-{task}-{dataset_part}-bin"
    else:
        data_dir /= f"{dataset}-{task}-bin"

    if task == "mlm":
        data_dir = data_dir / "input0"

    if not data_dir.exists():
        raise ValueError(f"Can't find Processed Dataset!\n"
                         f"Check your path: {data_dir}")

    EXT_VARS["EXPERIMENT_DIR"] = str(experiment_dir)
    EXT_VARS["OUTPUT_DIR"] = run_dir
    if task == "clf":
        roberta_dir, ckpt = sys.argv[2].split("=")[1].rsplit("/", maxsplit=1)
        
        ckpt = re.findall("checkpoint(_?\\d+(_\\d+)?).pt", ckpt)[0]
        EXT_VARS["OUTPUT_DIR"] = f"{Path(roberta_dir).parent}/{dataset}_clf_tune_ckpt_{ckpt}"
    config = evaluate_file(config_path,
                           ext_vars=EXT_VARS,
                           native_callbacks=CALLBACKS)
    config = json.loads(config)
    base_config = config.pop("base_config")
    task_hparams = json.loads(evaluate_file(hparams_config_path))
    config = {**base_config, **task_hparams, **config}

    if not Path(config["restore_file"]).exists():
        raise ValueError(f"Can't find RoBERTa checkpoint!\n"
                         f"Check your path: {config['restore_file']}")
    
    if task == 'mlm':
        (Path(run_dir) / "ckpt").mkdir(exist_ok=True, parents=True)
        copyfile(config["restore_file"],  f"{run_dir}/ckpt/checkpoint0.pt")
                         
    extra_arg_names = []
    for arg in sys.argv[2:]:
        argname = arg.split("=")[0][2:]
        if argname == "dataset_part":
            continue
        
        extra_arg_names.append(argname)
    extra_arg_names = set(extra_arg_names)

    config_args = [str(data_dir)]
    for argname, argvalue in config.items():
    
        cur_name = argname.replace('_', '-')
        if(cur_name in extra_arg_names):
            continue
            
        new_arg = f"--{argname.replace('_', '-')}"
        if not isinstance(argvalue, bool):
            new_arg += f"={argvalue}"
        config_args.append(new_arg)
    train_args = sys.argv[:1] + config_args
    
    for arg in sys.argv[2:]:
        argname = arg.split("=")[0][2:].replace("-", "_")
        if argname == "dataset_part":
            continue
        if argname == "path_to_scores":
             argvalue = arg.split("=")[1]
             argvalue = experiment_dir/"SCORES"/argvalue
             arg = arg.split('=')[0]+'='+str(argvalue)
        #if argname not in config:
        train_args.append(arg)

    sys.argv = train_args
    train()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True, type=str)
    parser.add_argument("--dataset-part", required=False, type=str, default="")
    known_args, _ = parser.parse_known_args()
    main(known_args)
