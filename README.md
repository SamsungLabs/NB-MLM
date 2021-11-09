# NB-MLM

This is code for the paper:
Nikolay Arefyev, Dmitrii Kharchev, Artem Shelmanov. NB-MLM: Efficient Domain Adaptation of Masked Language Models for Sentiment Analysis
https://aclanthology.org/2021.emnlp-main.717/

## Installation
Clone NB-MLM repository from github.com.
```shell script
git clone https://github.com/TODO/taskadapt/
cd taskadapt/ROBERTA
```

### Setup anaconda environment alternative
1. Download and install [conda](https://conda.io/docs/user-guide/install/download.html)
2. Create new conda environment and install dependencies
```shell script
   conda env create -f environment.yml
```
3. Activate environment
```shell script
   conda activate nbmlm
```

### Setup fairseq right version
1. Clone fairseq repository from github.com.
    ```shell script
    git clone https://github.com/pytorch/fairseq
    cd fairseq
    ```
2. Set repository to version 0.10.1 and install it
    ```shell script
    git checkout 8a0b56e    
    pip install --editable ./
    ```
3. Return to the main dir
     ```shell script
    cd ./../
     ```

### Download RoBERTa pretrained
```shell script
    bash install_roberta.sh
```



## Results

<table>
    <thead>
        <tr>
            <th rowspan=2><b>Model</b></th>
            <th colspan=2><b>Imdb</b></th>
            <th colspan=2><b>Yelp. P</b></th>
        </tr>
        <tr>
            <th>ERR</th>
            <th>macro-F1</th>
            <th>ERR</th>
            <th>macro-F1</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Uniform DAPT+all-TAPT</td>
            <td>3.62</td>
            <td>96.38</td>
            <td>1.55</td>
            <td>98.45</td>
        </tr>
        <tr>
            <td>NB-MLM DAPT+all-TAPT</td>
            <td>3.54</td>
            <td>96.46</td>
            <td>1.51</td>
            <td>98.49</td>
        </tr>
    </tbody>     
</table>


## Data preparation


1. Download Amazon, imdb, yelp datasets

    ```shell script
        bash reproduction_download_all.sh
    ```

2. Convert imbd and yelp to fairseq format and generate Naive Bayes weighted scores for nb-mlm. These scripts can be executed in parallel

    ```shell script
        bash reproduction_prepare_imdb_data_scores.sh
        bash reproduction_prepare_yelp_data_scores.sh
    ```
3. Split amazon on train, dev; convert amazon to fairseq format; assembly directories for DAPT.
   ```shell script
        cd ./amazon_new
        bash prepare_amazon_data.sh
        bash make_amazon_data_dir.sh
        bash make_amazonYelp_data_dir.sh
        cd ./../
    ```

## Before reproduce read this

We conducted all experiments using only one GPU. If your batch does not fit on one gpu, then you can use several weaker GPUs or change the relation between batch-size and gradient-accumulation-steps.
It must be remembered that an effective size of the batch is considered in fairseq like batch-size\*gradient-accumulation-steps\*number-of-gpu.

You can set a number of visible gpu with CUDA_VISBLE_DEVICE.

You can change batch-size and gradient-accumulation-steps in ./configs/"$DATASET_NAME"/mlm-all-hparams.jsonnet for mlm

You can change batch-size and gradient-accumulation-steps in ./configs/"$DATASET_NAME"/clf-hparams.jsonnet for classifier

## imdb reproduction

### if you want to reproduce long Uniform DAPT+TAPT, then run:

```shell script
    bash reproduction_uniform_imdb.sh
```

### if you want to reproduce long NB-MLM DAPT+TAPT, then run:

```shell script
    bash reproduction_nb-mlm_imdb.sh
```

### Where you can find results:

for Uniform

Metrics
```shell script
nano ./imdb_experiments/RUNS/imdb_all_uniform_mlm_seed_103_restore_file_./imdb_clf_tune_ckpt_('90', '')/ckpt/predicted_test.metrics
```
Labels
```shell script
nano ./imdb_experiments/RUNS/imdb_all_uniform_mlm_seed_103_restore_file_./imdb_clf_tune_ckpt_('90', '')/ckpt/predicted_test.labels
```

for NB-MLM
Metrics
```shell script
nano ./imdb_experiments/RUNS/imdb_all_nb_mlm_temperature_0.4_path_to_scores_m_50_seed_101_restore_file_./imdb_clf_tune_ckpt_('90', '')/ckpt/predicted_test.metrics
```
Labels
```shell script
nano ./imdb_experiments/RUNS/imdb_all_nb_mlm_temperature_0.4_path_to_scores_m_50_seed_101_restore_file_./imdb_clf_tune_ckpt_('90', '')/ckpt/predicted_test.labels
```


## Yelp reproduction

### if you want to reproduce long Uniform DAPT+TAPT, then run:

```shell script
bash reproduction_uniform_yelp.sh
```

### if you want to reproduce long NB-MLM DAPT+TAPT, then run:

```shell script
bash reproduction_nb-mlm_yelp.sh
```


### Where you can find results:

for Uniform

Metrics
```shell script
nano ./yelp_experiments/RUNS/yelp_all_uniform_mlm_seed_2021_restore_file_./yelp_clf_tune_ckpt_('27', '')/ckpt/predicted_test.metrics
```
Labels
```shell script
nano ./yelp_experiments/RUNS/yelp_all_uniform_mlm_seed_2021_restore_file_./yelp_clf_tune_ckpt_('27', '')/ckpt/predicted_test.labels
```

for NB-MLM

Metrics
```shell script
nano ./yelp_experiments/RUNS/yelp_all_nb_mlm_temperature_0.4_path_to_scores_m_50_seed_2020_restore_file_./yelp_clf_tune_ckpt_('27', '')/ckpt/predicted_test.metrics
```
Labels
```shell script
nano ./yelp_experiments/RUNS/yelp_all_nb_mlm_temperature_0.4_path_to_scores_m_50_seed_2020_restore_file_./yelp_clf_tune_ckpt_('27', '')/ckpt/predicted_test.labels
```




