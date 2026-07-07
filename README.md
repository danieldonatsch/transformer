# Transformers

Here I put the code I wrote, copied, cloned, stole or used in any other way to get to know transformers, vision transformers, LLMs, VLMs, etc.
What I have so far is a classic (word/language based) transformer and a vision transformer.

## Transformer and Language Models

A transformer and an LLM-like structure can be found in the following files:
- `transformer.py`, the transformer architecture consisting of:
- `attention.py`, multi-headed attention and
- `multilayer_perceptron.py`, the MLP. These are wrapped in a
- `mini_language_model.py`, and use 
- `position_encoding.py` for position encoding. With
- `train_and_run_MLM.py` is the model trained and tested.

To train it (and test) it, I used the Wikipedia article about Large Language Models.
The raw data can be found in `data/Wikipedia_LLM_orig.txt`.
I cleand it then with `data_cleaning.py` and it is then loaded to the model with `data_loader.py`.
It is not enough data for getting a good result.
But it helped to understand (and test the understanding) of the explanation from [3 blue 1 brown](https://www.3blue1brown.com/?topic=neural-networks).