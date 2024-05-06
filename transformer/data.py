"""
Get dataset. For the Tiny Shakespeare dataset, one token = one character.
For the OpenWebText dataset, the GPT2 tokenizer is used.
"""

import json
import os
import shutil
import urllib
from typing import Dict, List, Optional
from warnings import warn

import datasets
import numpy as np
import tiktoken

from options import get_parser__data_prep


def encode(text: str, vocab: List[str]) -> List[int]:
    """
    Encode the text.

    Args:
        text: Text to encode.
        vocab: Vocabulary.

    Returns:
        Encoded text.
    """
    string_to_int = {char: idx for (idx, char) in enumerate(vocab)}
    return [string_to_int[char] for char in text]


def decode(tokens: List[int], vocab: List[str]) -> List[str]:
    """
    Decode the text.

    Args:
        text: Text to decode.
        vocab: Vocabulary.

    Returns:
        Decoded text.
    """
    int_to_string = {idx: char for (idx, char) in enumerate(vocab)}
    return "".join([int_to_string[idx] for idx in tokens])


def save_shakespeare(train_split: int) -> None:
    """
    Save the Shakespeare dataset as well as meta data to disk.

    Args:
        train_split: Fraction of the dataset to use for training.
    """
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    # https://stackoverflow.com/a/7244263
    with urllib.request.urlopen(url) as response, open(
        "input.txt", "wb"
    ) as out_file:
        shutil.copyfileobj(response, out_file)

    with open("input.txt", "r") as f:
        text = f.read()
    os.remove("input.txt")

    # get vocabulary and its size
    vocab = sorted(list(set(text)))
    vocab_size = len(vocab)

    # tokenize the text
    data = np.array(encode(text, vocab=vocab), dtype=np.int64)
    print(
        f"Length of dataset in characters: {len(text)}\nFirst 100 "
        f"characters of dataset: {text[:100]}\nVocabulary size: "
        f"{vocab_size}\nVocabulary: {' '.join(vocab)}\n"
        f"Encoded text: {data}; shape: {data.shape}, dtype: {data.dtype}"
    )

    # make train-val split
    num_train_examples = int(train_split * len(data))
    train_data = data[:num_train_examples]
    val_data = data[num_train_examples:]

    # save train/val data and metadata
    np.save("train_data_shakespeare.npy", train_data)
    np.save("val_data_shakespeare.npy", val_data)
    meta_data = {"vocab": vocab, "vocab_size": vocab_size}
    with open("meta.json", "w") as f:
        json.dump(meta_data, f)


def save_openweb(train_split: int, num_proc: Optional[int] = None) -> None:
    """
    Save the OpenWebText dataset.

    Args:
        train_split: Fraction of the dataset to use for training.
        num_proc: Number of processes when downloading and generating the
            dataset locally. Multiprocessing is disabled by default.
    """
    split_dataset = datasets.load_dataset(
        "openwebtext", split=datasets.Split.TRAIN, num_proc=num_proc
    ).train_test_split(
        test_size=1 - train_split, seed=0, shuffle=True
    )  # `DictDataset`

    # rename test to val dataset:
    split_dataset["val"] = split_dataset.pop("test")

    print(f"Data type: {split_dataset['train']}")

    # define BPE tokenizer
    bpe_tokenizer = tiktoken.get_encoding("gpt2")

    def process(tokenizer: tiktoken.core.Encoding, example: Dict) -> Dict:
        """
        Encode text and return length of tokens.

        Args:
            example: Example to process.

        Returns:
            Processed example.
        """
        # ignore special tokens:
        token_ids = tokenizer.encode_ordinary(example["text"])
        # add end-of-sentence token:
        token_ids.append(bpe_tokenizer.eot_token)

        return {"token_ids": token_ids, "len": len(token_ids)}

    # tokenize the train and val dataset
    dataset_dict = split_dataset.map(
        function=lambda x: process(bpe_tokenizer, x),
        remove_columns=["text"],  # to reduce memory
        num_proc=num_proc,
    )  # `DatasetDict`, keys: "train", "val"

    # concatenate all tokenized ids in each dataset into one large file to
    # be used for training/validation:
    for split, dset in dataset_dict.items():
        # total number of tokens over all sentences summed:
        arr_len = np.sum(dset["len"], dtype=np.int64)
        array = np.memmap(
            filename=f"{split}_data_openweb.npy",
            dtype=np.int64,
            mode="w+",
            shape=(arr_len,),
        )

        total_batches = 1024
        idx = 0
        for batch_idx in range(total_batches):
            # Batch together samples for faster write
            batch = dset.shard(
                num_shards=total_batches, index=batch_idx, contiguous=True
            ).with_format("numpy")
            array_batch = np.concatenate(batch["token_ids"])
            # Write into mmap
            array[idx : idx + len(array_batch)] = array_batch
            idx += len(array_batch)
        # flush `memmap` instance to disk to write changes to file
        # https://numpy.org/doc/stable/reference/generated/numpy.memmap.html
        array.flush()


def main(
    train_split: int = 0.8,
    dataset: str = "shakespeare",
    num_proc: Optional[int] = None,
) -> None:
    """
    Save the chosen dataset (and some meta data for the Shakespeare dataset)
    to disk.

    Args:
        train_split: Fraction of the dataset to use for training.
        dataset: Dataset to use. Options are 'shakespeare' and 'openweb'.
        num_proc: Number of processes when downloading and generating the
            openweb dataset locally. Multiprocessing is disabled by default.
    """
    assert (
        0 < train_split < 1
    ), f"Train split should be > 0 and < 1, but is instead {train_split}"
    assert dataset in [
        "shakespeare",
        "openweb",
    ], f"Dataset should be 'shakespeare' or 'openweb', but is '{dataset}'."

    if dataset == "shakespeare" and num_proc is not None:
        warn("Argument `num_proc` is ignored for the Shakespeare dataset.")

    if dataset == "shakespeare":
        save_shakespeare(train_split=train_split)
    else:
        save_openweb(train_split=train_split, num_proc=num_proc)


if __name__ == "__main__":
    parser = get_parser__data_prep()
    args = parser.parse_args()
    print(f"Flags:\n{args}\n")

    main(
        train_split=args.train_split,
        dataset=args.dataset,
        num_proc=args.num_proc,
    )
