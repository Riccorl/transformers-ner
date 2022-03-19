# Transformers NER

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1X6zEbRV0sZzcZCVC3Ir2j3TXEUwC0hL-?usp=sharing)
[![PyTorch](https://img.shields.io/badge/PyTorch-orange?logo=pytorch)](https://pytorch.org/)
[![Transformer-Embedder](https://img.shields.io/badge/Transformers%20Embedder-2.0.0-6670ff)](https://github.com/Riccorl/transformers-embedder)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000)](https://github.com/psf/black)

This is an example project for [Transformers Embedder](https://github.com/Riccorl/transformers-embedder) library.

First thing first:

```bash
./scripts/setup.sh
```

## Train

To train a model use:

```bash
./scripts/train.sh "${CONFIG_NAME}"
```

where `${CONFIG_NAME}` is the name of one of the `yaml` file in `conf` folder, e.g. `bert_base`.

The main parameters available are

- `language_model_name`: a language model name/path from HuggingFace transformers library
- `model_name`: the name of the experiment
- `layer_pooling_strategy`: from `transformers-embedder`, what kind of output the transformers should give
- `return_words`: from `transformers-embedder`, whether to perform sub-word pooling to get back words

There are other experiment related parameters in the files in `conf` directory.

## Evaluation

Run the following code to evaluate the model against the test set:

```bash
python src/evaluate.py \
  --config-name="${CONFIG_NAME}" \
  evaluate.checkpoint_path="/absolute/path/to/checkpoint"
```

## Results

These models are trained for 10 epochs, using RAdam with a learning rate of `1e-5`.

| Language Model 	| F1   	| Inference Time (GPU) 	|
|----------------	|------	|----------------------	|
| MiniLM         	| 90.0 	|          6s          	|
| DistilBERT     	| 88.9  |          6s          	|
| BERT-base      	|      	|                      	|
| RoBERTa-large   | 91.4 	|          24s         	|
