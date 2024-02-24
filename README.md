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

## TODO
[ ] write class `Transformer` that combines `Encoder` and `Decoder`
[ ] implement `mask`: https://peterbloem.nl/blog/transformers
[ ] add `check_args` function
[ ] add github action workflows
```
mask = torch.ones(
    seq_length, seq_length
).tril()
```
which will get broadcasted in the classes `EncoderBlock` and `DecoderBlock`.
[ ] is the way of retrieving the attention maps in the `Encoder` class correct?
