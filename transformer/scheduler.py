from torch.optim import Optimizer


class LRScheduler:
    """
    Class to implement lr scheduling of [1].

    [1] http://arxiv.org/abs/1706.03762
    """

    def __init__(
        self,
        optimizer: Optimizer,
        d_model: int,
        warmup_steps: int,
        lr_multiplier: float = 1,
    ) -> None:
        """
        Initialize the learning rate scheduler.

        Args:
            optimizer: Optimizer.
            lr_multiplier: By which factor to multiply the learning rates
                of the scheduler used in the 'Attention is All You Need Paper'.
            d_model: Embedding dimension.
            warmup_steps: Number of warmup steps (during which learning rate
                is increased).
        """
        self.optimizer = optimizer
        self.lr_multiplier = lr_multiplier
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def _update_lr(self) -> None:
        """
        Update the learning rate, cf. Eq. (3) of [1].
        """
        lr = (
            self.lr_multiplier
            * (self.d_model**-0.5)
            * min(
                self.step_num**-0.5,
                self.step_num * self.warmup_steps**-1.5,
            )
        )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def step(self) -> None:
        """
        Take a step in the lr schedule (i.e. update the learning rate and take
        an optimization step).
        """
        self.step_num += 1
        self._update_lr()
