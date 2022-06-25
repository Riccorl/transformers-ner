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
./scripts/train.sh "${LANGUAGE_MODEL_NAME}"
```

where `${LANGUAGE_MODEL_NAME}` is the name of one of the language models supported by HuggingFace, e.g. 
`bert-base-cased`. You can look at the `conf` directory to see the list of supported hyper-parameters.

## Evaluation

Run the following code to evaluate the model against the test set:

```bash
python src/evaluate.py \
  "model.model.language_model=$LANGUAGE_MODEL_NAME" \
  evaluate.checkpoint_path="/absolute/path/to/checkpoint"
```

## Results

These models are trained for 10 epochs, using RAdam with a learning rate of `1e-5`.

| Language Model 	| F1   	| Inference Time (GPU) 	|
|----------------	|------	|----------------------	|
| MiniLM         	| 90.0 	|          6s          	|
| M-MiniLM        | 88.2 	|          6s          	|
| DistilBERT     	| 88.9  |          6s          	|
| BERT-base      	| 90.1 	|                      	|
| RoBERTa-large   | 91.4 	|          24s         	|
