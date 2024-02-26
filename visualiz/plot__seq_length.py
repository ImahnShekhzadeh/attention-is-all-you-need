"""
Plot the sequence lengths (over train, val and test concatenatted) of the 
sentences in the the IWSLT2017 DE-EN dataset for a fixed vocabulary size.
"""
import logging
import os

import argparse
from datetime import datetime as dt
import numpy as np
from datasets import load_dataset
from matplotlib import pyplot as plt, ticker
from tokenizers import Tokenizer

from utils import check_args, get_len_tokenized_data

def main(args: argparse.Namespace) -> None:
    """Main function."""

    # Setup basic configuration for logging
    logging.basicConfig(
        filename="dataset.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    # load dataset
    # type: `datasets.dataset_dict.DatasetDict`
    data = load_dataset("iwslt2017", "iwslt2017-de-en")

    # load tokenizer with fixed vocab length
    tokenizer = Tokenizer.from_file(args.tokenizer_file)
    logging.info(
        f"=> Tokenizer `{args.tokenizer_file}` with a vocabulary size of "
        f"{tokenizer.get_vocab_size()} loaded."
    )

    # get token lengths (sequence lengths)
    token_lengths = get_len_tokenized_data(tokenizer=tokenizer, data=data)

    # plot frequency
    bins = np.arange(min(token_lengths) - 1, max(token_lengths) + 2, 1) - 0.01

    plt.hist(token_lengths, bins=bins, align="left")
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=1))
    plt.xlabel("Sequency length")
    plt.ylabel("Frequency")
    plt.savefig(
        f"seq_lengths_{dt.now().strftime('%dp%mp%y_%Hp%Mp%S')}.pdf"
    )
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyperparameters and parameters"
    )
    parser.add_argument(
        "--tokenizer_file",
        required=True,
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