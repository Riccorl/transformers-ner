# Transformers NER

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1X6zEbRV0sZzcZCVC3Ir2j3TXEUwC0hL-?usp=sharing)
[![PyTorch](https://img.shields.io/badge/PyTorch-orange?logo=pytorch)](https://pytorch.org/)
[![Transformer-Embedder](https://img.shields.io/badge/Transformers%20Embedder-3.0.2-6670ff)](https://github.com/Riccorl/transformers-embedder)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000)](https://github.com/psf/black)

This is an example project for [Transformers Embedder](https://github.com/Riccorl/transformers-embedder) library.

First thing first:

```bash
./scripts/setup.sh
```

## Train

To train a model use:

```bash
./scripts/train.sh -l "${LANGUAGE_MODEL_NAME} ${OVERRIDES}"
```

where `${LANGUAGE_MODEL_NAME}` is the name of one of the language models supported by HuggingFace, e.g. 
`bert-base-cased`. You can look at the `conf` directory to see the list of supported hyper-parameters.
`${OVERRIDES}` instead, can be used to override the hydra configuration files, using the hydra syntax, e.g.
`model.model.subword_pooling_strategy=scatter`:

```bash
./scripts/train.sh -l bert-base-cased "model.model.subword_pooling_strategy=scatter"
```

A usage description of the `train.sh` is provided below.

```bash
train.sh [-h] [-l LANG_MODEL_NAME] [-d] [-p PRECISION] [-c] [-g DEVICES] [-n NODES] [-m GPU_MEM] [-s STRATEGY] [-o] OVERRIDES

where:
    -h            Show this help text
    -l            Language model name (one of the models from HuggingFace)
    -d            Run in debug mode (no GPU and wandb offline)
    -p            Training precision, default 16.
    -c            Use CPU instead of GPU.
    -g            How many GPU to use, default 1. If 0, use CPU.
    -n            How many nodes to use, default 1.
    -m            Minimum GPU memory required in MB (default: 8000). If less that this,
                  training will wait until there is enough space.
    -s            Strategy to use for distributed training, default NULL.
    -o            Run the experiment offline
    OVERRIDES     Overrides for the experiment, in the form of key=value.
                  For example, 'model_name=bert-base-uncased'
Example:
  ./script/train.sh -l bert-base-cased
  ./script/train.sh -l bert-base-cased -m 10000
```
 
## Evaluation

Run the following code to evaluate the model against the test set:

```bash
python src/evaluate.py \
  "model.model.language_model=$LANGUAGE_MODEL_NAME" \
  evaluate.checkpoint_path="/absolute/path/to/checkpoint"
```

## Results

### CoNLL 2003

These models are trained for 10 epochs, using RAdam with a learning rate of `1e-5`.

| Language Model 	| F1   	| Inference Time (GPU) 	|
|----------------	|------	|----------------------	|
| MiniLM         	| 90.0 	|          6s          	|
| M-MiniLM        | 88.2 	|          6s          	|
| DistilBERT     	| 88.9  |          6s          	|
| BERT-base      	| 90.1 	|                      	|
| RoBERTa-large   | 91.4 	|          24s         	|

### CoNLL 2012