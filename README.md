# Rethinking Regularization Methods for Knowledge Graph Completion (SPR)

This repository contains the reference implementation for SPR proposed in our paper **Rethinking Regularization Methods for Knowledge Graph Embedding**. 




---

## Directory Layout

```
SPR/
├── Baseline/                 # third‑party baseline models (CompGCN, GIE, HGE)
├── logs/                     # tensorboard summaries, checkpoints, configs
│   └── CP_SPR_WN18RR_0/      # example run folder
├── model/                    # SPR implementation
│   ├── datasets.py           # dataset wrappers (FB237, WN18RR …)
│   ├── models.py             # model zoo + SPR regulariser
│   ├── optimizers.py         # Ranger, AdamW, Adagrad, etc.
│   ├── regularizers.py       # **<‑‑ SPR lives here**
│   ├── learn.py              # training loop
│   ├── process_datasets.py   # converts raw triples to numeric tensors
│   └── run.bash              # entry‑point script
├── src_data/                 # raw datasets (place yours here)
│   ├── FB237/  kinships/  UML/  WN18RR/  YAGO3‑10/
├── requirements.txt          # python dependencies
└── README.md                 # you are here
```

---

## Installation

We recommend **Python 3.9+** and **CUDA 11.4+**.

```bash

# create env (optional but recommended)
$ conda create -n spr python=3.9 && conda activate spr

# install all python packages
$ pip install -r requirements.txt
```

All experiments in the paper were run on **8 × NVIDIA V100‑32GB GPUs** (multi‑GPU training via `torch.distributed`).  SPR also runs on a single GPU/CPU for small‑scale tests.

---

## Data Preparation

1. **prepare your dataset** (train/valid/test triples in TSV format – *head \t relation \t tail*) into `src_data/<DATASET_NAME>/`.
2. Execute the preprocessing script:

   ```bash
   python model/process_datasets.py --data_dir src_data/<DATASET_NAME>
   ```



---

## Training & Evaluation
#### To facilitate the work of reviewers, we provide a very simple one-stop run shell script, you can run it like this to reproduce our results(Recommended options):

> ```bash
> pip install -r requirements.txt            # install dependencies
> python model/process_datasets.py           # preprocess any dataset in src_data/
> bash model/run.bash                        # train & evaluate with default SPR config
> ```
---

#### Or use this code with a more sophisticated code command:

The easiest way to run an experiment is via the provided shell script:

```bash
bash model/run.bash WN18RR 0   # <DATASET> <GPU_ID>
```

`run.bash` is a thin wrapper around `learn.py`.  You can override any hyper‑parameter from the command line, e.g.:

```bash
python model/learn.py \
  --dataset FB237 \
  --model ... \
  --batch_size ... \
  --lr ... \
  --gpu 0
  .....
```




---



## Logging & Checkpoints

* **Config** – every run stores the exact CLI/JSON configuration in `logs/<RUN_NAME>/config.json`.
* **TensorBoard** – open with `tensorboard --logdir logs/` to inspect loss & MRR curves.
* **Model Weights** – best checkpoint (`*.ckpt`) is saved when validation MRR improves.

Resume training with:

```bash
python model/learn.py --resume logs/CP_SPR_WN18RR_0 --gpu 0
```

---






## Citation



---



---

### Acknowledgements
We are grateful for the following excellent baseline models and methods:

1. **CompGCN:** Vashishth et al., *ICLR 2020*
2. **GIE:** Han et al., *AAAI 2022*
3. **HGE:** Chami et al., *AAAI 2024*
6. **VIR:** Xiao et al., *NeurIPS 2024*
4. **DURA:** Zhang et al., *NeurIPS 2020,TPAMI 2022*
5. **ER:** Cao et al., *AAAI 2022*
...
and many other contributors within the community.

