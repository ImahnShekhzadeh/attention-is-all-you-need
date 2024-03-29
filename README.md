# attention-is-all-you-need
Implementation of the "Attention is All You Need" (arXiv:1706.03762) Paper

## Tokenizer
Note that the tokenizer `bpe_tokenizer_37k.json` used here was obtained by training on the training, validation and test data of the IWSLT2017 DE-EN data. This is the same data that the transformer sees. However, in this [video](https://www.youtube.com/watch?v=zduSFxRajkE), Karpathy recommends to use different data, so you might want to do that.

Also, please note that the BPE tokenizer was first trained specifying the special tokens `[UNK]`, `[CLS]`, `[SEP]`, `[PAD]` and `MASK`. Later, I modified my code such that only `[UNK]` and `[PAD]` are special tokens appearing in the vocabulary, and added `[SOS]` as a start of sentence token. To avoid having to retrain the tokenizer for about two days on a consumer-grade CPU, I modified the used vocabulary [`bpe_tokenizer_37k.json`](transformer/bpe_tokenizer_37k.json) by removing the special token `[CLS]` and usnig `[SOS]` instead. This modification happened in-place.

## Run
To run the python script,
```
docker build -f Dockerfile -t transformers:1.0.0 .
docker run --shm-size 512m --rm -v $(pwd):/app --gpus all -it transformers:1.0.0 --config configs/conf.json --train
```
In order to generate text from a pre-trained model, run
```
docker run --shm-size 512m --rm -v $(pwd):/app --gpus all -it transformers:1.0.0 --config configs/conf.json --loading_path <loading_path>
```
To run the python script locally in a conda environment, which allows for faster debugging,
```
python3 -B transformer/run.py --config configs/conf.json
```
As of 08 February 2024, the stable version of `torchtext` is `0.17.0`, which does not have the WMT 2024 EN-DE dataset included, which the original [Attention is All You Need](http://arxiv.org/abs/1706.03762) paper used. Hence, the IWSLT2017 DE-EN dataset is used.

In the original [Attention is All You Need](http://arxiv.org/abs/1706.03762) paper, the model is reported to have about `65`M parameters, the implemented transformer in this repo - with the options provided in `configs/conf.json` - has about `63`M params.

### W&B
If you want to log some metrics to [Weights & Biases](https://wandb.ai/), append the following to the `docker run` command:
```
--wandb__api_key <api_key>
# --wandb__api_key 2fru...
```

## Sequence Length
To justify the chosen sequence length, I wrote a script that - for a given pre-trained tokenizer - plots the frequency of tokens vs their number.
Run the script as follows,
```
cd visualiz/
python3 -B plot__seq_length.py --tokenizer_file ...
```
You can either use my pre-trained tokenizer, or provide your own. For the used dataset and the BPE tokenizer `transformer/bpe_tokenizer_37k.json` with a vocabulary size of `37`k, here is the plot:

<div style="display: flex; justify-content: center;">
    <img src="visualiz/seq_lengths_37k.png" alt="Description" width="600"/>
</div>
<br>

For the chosen BPE tokenizer, the total number of *train* tokens is `8.8`M.

## TODO
[ ] in `models.py`, create the "look-ahead" mask **outside** the forward func and instead add arg `tgt_mask`
[ ] introduce args `tgt_padding_mask` and `src_padding_mask` in forward func of Transformer
[ ] in `scaled_dot_product_attn()`, refactor signature: `mask` and `padding_mask`. if both are provided, then `torch.min(mask, padding_mask)` should be the `mask` applied.
[ ] add start token to encoder input as well
[ ] add flag to specify # checkpoints, implement code for this
[ ] add label smoothing
[ ] add github action workflows
