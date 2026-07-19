# Transformers

Here I put the code I wrote, copied, cloned, stole or used in any other way to get to know transformers, LLMs and vision transformers.
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

## Vision Transformer

Once the transformer-basics are understood, I wanted to understand also the vision-transformer.
So, I dived into this as well.
I followed to large parts the Visuara AI [YouTube Video](https://www.youtube.com/watch?v=DdsVwTodycw)
and [Medium Article](https://medium.com/@vizuara/coding-a-vision-transformer-almost-from-scratch-178545e85ada).
I tried to stay consistent with the naming conventions from [3 blue 1 brown](https://www.3blue1brown.com/?topic=neural-networks)
and re-used also some of the code written earlier:
- `vision_transformer.py`, the transformer architecture consisting of:
- `patch_embedding.py`, where the image is split in blocks and embeddings are built and
- `vit_block.py`, the actual transformer block. That one consists again of two sub-blocks:
  - multi-headed attention, where we use the PyTorch implementation (instead of ours) and
  - `position_encoding.py`, code re-used from the text transformer above. Finally,
- `train_and_run_ViT.py` is the script to train and test the model.

We use the MNIST data-set, therefore no data cleaning or similar is needed.
