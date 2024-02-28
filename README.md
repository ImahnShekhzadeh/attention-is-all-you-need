# attention-is-all-you-need
Implementation of the "Attention is All You Need" (arXiv:1706.03762) Paper

## Notes
Note that the tokenizer `bpe_tokenizer_37k.json` used here was obtained by training on the training, validation and test data of the IWSLT2017 DE-EN data. This is the same data that the transformer sees. However, in this [video](https://www.youtube.com/watch?v=zduSFxRajkE), Karpathy recommends to use different data, so you might want to do that.

## Run
To run the python script,
```
docker build -f Dockerfile -t transformers:1.0.0 .
docker run --shm-size 512m --rm -v $(pwd):/app --gpus all -it transformers:1.0.0 --config configs/conf.json
```
To run the python script locally in a conda environment, which allows for faster debugging,
```
python3 -B transformer/run.py --config configs/conf.json
```
As of 08 February 2024, the stable version of `torchtext` is `0.17.0`, which does not have the WMT 2024 EN-DE dataset included, which the original [Attention is All You Need](http://arxiv.org/abs/1706.03762) paper used. Hence, the IWSLT2017 DE-EN dataset is used.

In the original [Attention is All You Need](http://arxiv.org/abs/1706.03762) paper, the model is reported to have about `65`M parameters, the implemented transformer in this repo - with the options provided in `configs/conf.json` - has about `63`M params.

## Sequence Length
To justify the chosen sequence length, I wrote a script that - for a given pre-trained tokenizer - plots the frequency of tokens vs their number.
Run the script as follows,
```
cd visualiz/
python3 -B plot__seq_length.py --tokenizer_file ...
```
You can either use my pre-trained tokenizer, or provide your own. For the used dataset and the BPE tokenizer `transformer/bpe_tokenizer_37k.json` with a vocabulary size of `37`k, you can iew the plot [here](visualiz/seq_lengths_37k.pdf). For the chosen BPE tokenizer, the total number of *train* tokens is `8.8`M.

## TODO
[ ] add github action workflows
[ ] remove flag `learning_rate` and instead add flag `warmup_steps`
