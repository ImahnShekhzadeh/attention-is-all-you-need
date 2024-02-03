import torch

class LRScheduler:
    """
    Class to implement lr scheduling of [1].

    [1] http://arxiv.org/abs/1706.03762
    """
    def __init__(self, d_model, warmup_steps) -> None:
        """
        Args:
            d_model: dimension of the model
            warmup_steps: int, number of warmup steps
        """
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0