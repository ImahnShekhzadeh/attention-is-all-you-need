"""
Utility functions/classes for the script `plot__seq_length.py`.
"""
from typing import List

import datasets
from tokenizers import Tokenizer


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

    token_lengths = []

    for split in ["train", "validation", "test"]:  # TODO: try `data.keys()`
        for dict in data[split]["translation"]:
            token_lengths.append(len(tokenizer.encode(dict["de"]).ids))
            token_lengths.append(len(tokenizer.encode(dict["en"]).ids))

    return token_lengths
