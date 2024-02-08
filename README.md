# attention-is-all-you-need
Implementation of the "Attention is All You Need" (arXiv:1706.03762) Paper

## Run
Opening the jupyter NB `transformers.ipynb` in the web,
```
docker build -f Dockerfile -t transformers:1.0.0 .
docker run -it --rm --gpus all -v $(pwd):/app -p 8888:8888 transformers:1.0.0
```
Then copy-paste one of the displayed URLs into the web browser.



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
