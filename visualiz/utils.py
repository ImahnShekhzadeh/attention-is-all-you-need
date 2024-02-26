"""
Utility functions/classes for the script `plot__seq_length.py`.
"""
import argparse
import os
from typing import List

import datasets
from tokenizers import Tokenizer


def check_args(args: argparse.Namespace) -> None:
    """
    Check provided arguments.
    """
    assert os.path.exists(
        args.tokenizer_file
    ), "Please provide a valid path to the tokenizer file."
    assert args.tokenizer_file.endswith(
        ".json"
    ), "If a tokenizer is provided, it must be a JSON file."


def get_len_tokenized_data(
    tokenizer: Tokenizer, data: datasets.dataset_dict.DatasetDict
) -> List[int]:
    """
    Tokenize data and append the number of tokens to a list.
    This function is specifically written for the IWSLT 2017 dataset.

    Args:
        tokenizer: Pre-trained tokenizer.
        data: Dataset (both train, val and test split, as well as DE and EN
            source/target sentences)

    Returns:
        List containing token length.
    """

    token_ids = []

    for split in ["train", "validation", "test"]:
        for dict in data[split]["translation"]:
            token_ids.append(len(tokenizer.encode(dict["de"]).ids))
            token_ids.append(len(tokenizer.encode(dict["en"]).ids))

    return token_ids
