[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1X6zEbRV0sZzcZCVC3Ir2j3TXEUwC0hL-?usp=sharing)
[![PyTorch](https://img.shields.io/badge/PyTorch-orange?logo=pytorch)](https://pytorch.org/)
[![Transformer-Embedder](https://img.shields.io/badge/Transformer%20Embedder-1.7-6670ff)](https://github.com/Riccorl/transformer-embedder)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000)](https://github.com/psf/black)

# Transformers NER

This is an example project for [Transformer Embedder](https://github.com/Riccorl/transformer-embedder) library.

First thing first:

```bash
pip install -r requirements.txt
```

## Train

To train a model use:

```bash
python src/train.py \
  language_model_name="distilbert-base-cased" \
  model_name="distilbert-base-ner" \
  output_layer="sum" \
  subtoken_pooling="mean"
```

The parameters available are

- `language_model_name`: a language model name/path from HuggingFace transformers library
- `model_name`: the name of the experiment
- `output_layer`: from `transformer-embedder`, what kind of output the transformers should give
- `subtoken_pooling`: from `transformer-embedder`, what kind of poolinf to perform to get back words from sub-tokens

There are other experiment related parameters in the files in `conf` directory.

## Evaluation

Run the following code to evaluate the model against the test set:

```bash
python src/evaluate.py checkpoint_path="/absolute/path/to/checkpoint"
```

## Results

This model trained using `bert-base-cased` as language model achieves this scores against the CoNLL-2003 test set

```text
overall_precision: 0.8953856991898556
overall_recall: 0.9162011173184358
overall_f1: 0.9056738220361628
overall_accuracy: 0.9813177609231771
```

and the following scores on the dev set:

```text
overall_precision: 0.9419875567513032
overall_recall: 0.9530452534875808
overall_f1: 0.9474841437632135 
overall_accuracy: 0.9907299634186897
```
