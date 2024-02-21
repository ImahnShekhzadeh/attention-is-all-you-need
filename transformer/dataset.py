"""
Define the dataset class for the transformer model.
"""
from typing import Dict, List

import torch
from torch.utils.data import Dataset


class DictDataset(Dataset):
    """
    Dataset containing the source and target token ids in a dictionary.
    """

    def __init__(
        self,
        data: Dict[str, List[int]],
    ) -> None:
        """
        Initialize the dataset.

        Args:
            data: dictionary with the keys "src_ids" and "target_ids" and the
                corresponding values being lists of integers (token ids).
            embed_dim: Embedding dim, referred to as `d_model` in [1].
            max__seq_length: Maximum expected sequence length.
            vocab_size: Vocabulary size.

        [1] http://arxiv.org/abs/1706.03762
        """
        self.data = data

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            length of the dataset
        """
        return len(self.data["src_ids"])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get an item from the dataset.

        Args:
            idx: index of the item

        Returns:
            dictionary with the keys "source" and "target"
        """
        return {
            "source": self.data["src_ids"][idx],
            "target": self.data["target_ids"][idx],
        }
