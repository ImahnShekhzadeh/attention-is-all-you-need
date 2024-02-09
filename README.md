# attention-is-all-you-need
Implementation of the "Attention is All You Need" (arXiv:1706.03762) Paper

## Run
Opening the jupyter NB `transformers.ipynb` in the web,
```
docker build -f Dockerfile -t transformers:1.0.0 .
docker run -it --rm --gpus all -v $(pwd):/app -p 8888:8888 transformers:1.0.0
```
Then copy-paste one of the displayed URLs into the web browser.

To run the python script,
```
docker build -f Dockerfile_run -t transformers:1.0.0 .
docker run --shm-size 512m --rm -v $(pwd):/app --gpus all -it transformers:1.0.0
```
As of 08 February 2024, the stable version of `torchtext` is `0.17.0`, which does not have the WMT 2024 EN-DE dataset included, which the original [Attention is All You Need](http://arxiv.org/abs/1706.03762) paper used. Hence, the IWSLT2017 dataset (with the pairing EN-DE) is used.

## TODO
[ ] download dataset (write a function `get_dataset()`: https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/datasets/wmt.html)
[ ] implement `mask`: https://peterbloem.nl/blog/transformers
```
mask = torch.ones(
    seq_length, seq_length
).tril()
```
which will get broadcasted in the classes `EncoderBlock` and `DecoderBlock`.
[ ] is the way of retrieving the attention maps in the `Encoder` class correct?
