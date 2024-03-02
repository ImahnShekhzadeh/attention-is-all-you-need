"""
Plot the sequence lengths (over train, val and test concatenatted) of the
sentences in the the IWSLT2017 DE-EN dataset for a fixed vocabulary size.
"""
import argparse
from datetime import datetime as dt

import numpy as np
from datasets import load_dataset
from matplotlib import pyplot as plt
from matplotlib import ticker
from tokenizers import Tokenizer
from utils import check_args, get_len_tokenized_data


def main(args: argparse.Namespace) -> None:
    """Main function."""

    # load dataset
    # type: `datasets.dataset_dict.DatasetDict`
    data = load_dataset("iwslt2017", "iwslt2017-de-en")

    # load tokenizer with fixed vocab length
    tokenizer = Tokenizer.from_file(args.tokenizer_file)
    vocab_size = tokenizer.get_vocab_size()

    # get token lengths (sequence lengths)
    token_lengths = get_len_tokenized_data(tokenizer=tokenizer, data=data)

    # plot frequency
    bins = np.arange(min(token_lengths) - 1, max(token_lengths) + 2, 1) - 0.01

    plt.hist(token_lengths, bins=bins, align="left")
    ax = plt.gca()
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=2))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=10))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(base=int(1e3)))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=int(5e3)))
    plt.xlim(left=0)
    plt.xlabel("Sequence length (# tokens)")
    plt.ylabel("Frequency (# occurrence)")
    plt.savefig(
        f"seq_lengths_{vocab_size // 1000}k.png",
        bbox_inches="tight",
        pad_inches=0.01,
        dpi=600,
    )
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyperparameters and parameters"
    )
    parser.add_argument(
        "--tokenizer_file",
        type=str,
        default="../transformer/bpe_tokenizer_37k.json",
        help=(
            "Path to the tokenizer. If provided, the tokenizer will "
            "be loaded from this path."
        ),
    )
    args = parser.parse_args()

    check_args(args)

    main(args)
