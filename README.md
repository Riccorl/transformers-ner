# Transformers NER

This is an example project for [TransformerEmbedder](https://github.com/Riccorl/transformer-embedder) library.

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



## Using the repository
To use this repository as a starting template for you projects, you can just click the green button "Use this template" at the top of this page. More on using GitHub repositories on the following [link](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/creating-a-repository-from-a-template#creating-a-repository-from-a-template).
