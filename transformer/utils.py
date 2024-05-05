"""Utility functions and classes."""
import gc
import json
import logging
import os
import shutil
import sys
import urllib
from argparse import ArgumentParser, Namespace
from copy import deepcopy
from math import ceil
from time import perf_counter
from typing import Dict, Iterator, List, Optional, Tuple, Union

import datasets
import numpy as np
import tiktoken
import torch
import wandb
from prettytable import PrettyTable
from scheduler import LRScheduler
from torch import Tensor, autocast
from torch import distributed as dist
from torch import nn
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, DistributedSampler, IterableDataset

from architecture.attention import get_subsequent_mask
from data import decode, encode


def total_norm__grads(model: nn.Module) -> float:
    """
    Calculate total norm of gradients.

    Args:
        model: Model for which we want to check whether gradient clipping is
            necessary.
    """
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)  # in case 2-norm is clipped
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5

    return total_norm


def cleanup():
    """
    Cleanup the distributed environment.
    """
    dist.destroy_process_group()


def setup(rank: int, world_size: int, backend: str = "nccl"):
    """
    Initialize the distributed environment.

    Args:
        rank: Rank of the current process.
        world_size: Number of processes participating in the job.
        backend: Backend to use.
    """

    os.environ["MASTER_ADDR"] = "localhost"  # NOTE: might have to be adjusted
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )


def retrieve_args(parser: ArgumentParser) -> Namespace:
    """
    Retrieve and parse the args; some args might have been passed in a JSON
    config file.

    Returns:
        Argparse options.
    """
    args = parser.parse_args()

    if args.config is not None:
        if os.path.exists(args.config):
            assert args.config.endswith(".json"), (
                f"Config file should be a JSON file, but is a '{args.config}' "
                "file."
            )
            with open(args.config, "r") as f:
                config_args = json.load(f)  # type: dict

            # List all registered arguments
            registered_args = {action.dest for action in parser._actions}

            # Check if all config keys are known to the parser
            unknown_args = set(config_args) - registered_args

            # Check for unknown arguments
            if unknown_args:
                raise ValueError(
                    f"Unknown argument(s) in JSON config: {unknown_args}"
                )

            parser.set_defaults(**config_args)
            args = parser.parse_args()
        else:
            raise ValueError(f"Config file '{args.config}' not found.")

    check_args(args)

    return args


def check_args(args: Namespace) -> None:
    """
    Check provided arguments and print them to CLI.

    Args:
        args: Arguments provided by the user.
    """

    # create saving dir if non-existent
    os.makedirs(args.saving_path, exist_ok=True)

    assert args.compile_mode in [
        None,
        "default",
        "reduce-overhead",
        "max-autotune",
    ], (
        f"``{args.compile_mode}`` is not a valid compile mode in "
        "``torch.compile()``."
    )
    if args.pin_memory:
        assert args.num_workers > 0, (
            "With pinned memory, ``num_workers > 0`` should be chosen, cf. "
            "https://stackoverflow.com/questions/55563376/pytorch-how"
            "-does-pin-memory-work-in-dataloader"
        )
    assert 0 <= args.dropout_rate < 1, (
        "``dropout_rate`` should be chosen between 0 (inclusive) and 1 "
        f"(exclusive), but is {args.dropout_rate}."
    )
    if args.train:
        assert (
            args.num_steps > 0
        ), "Number of epochs should be greater than 0 when training."
    else:
        if args.num_steps > 0:
            logging.warning(
                "`--train` flag not set, but `--num_steps > 0` is provided."
                "Training will NOT be performed."
            )
        assert args.loading_path is not None, (
            "`--train` flag was not passed, so please provide a path to load "
            "the model from."
        )


def load_datasets(
    dataset: str,
) -> Tuple[Tensor, Tensor, int, Optional[List[str]]]:
    """
    Load Shakespeare or OpenWebText dataset.

    Args:
        dataset: Dataset to load. Should be 'shakespeare' or 'openweb'.
    
    Returns:
        Train and validation data, vocabulary, and vocabulary size.
        For the openwebtext dataset, the vocabulary is `None`.
    """
    if dataset == "shakespeare":
        train_data = torch.load("train_data_shakespeare.pt")
        val_data = torch.load("val_data_shakespeare.pt")
        with open("meta.json", "r") as f:
            meta_data = json.load(f)
        vocab_size = meta_data["vocab_size"]
        vocab = meta_data["vocab"]

    elif dataset == "openweb":
        bpe_tokenizer = tiktoken.get_encoding("gpt2")

        train_data = torch.load("train_data_openweb.pt")
        val_data = torch.load("val_data_openweb.pt")
        vocab_size = bpe_tokenizer.n_vocab
        vocab = None
    
    else:
        raise NotImplementedError(
            f"Dataset '{dataset}' not recognized. Please choose either "
            "'shakespeare' or 'openweb'."
        )
    
    return train_data, val_data, vocab_size, vocab


def get_batch(
    data: Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device | str | int = "cpu",
) -> Tuple[Tensor, Tensor]:
    """
    Get batch of data.

    Args:
        data: Train or validation data.
        batch_size: Batch size.
        block_size: Maximum context length.
        device: Device on which the code is executed.

    Returns:
        Batch of input in shape `(N, block_size)` and target tensors in shape
        `(N, block_size)`.
    """
    indices = torch.randint(
        low=0, high=len(data) - block_size, size=(batch_size,)
    )

    input = torch.stack(
        [data[idx : idx + block_size] for idx in indices], dim=0
    )
    target = torch.stack(
        [data[idx + 1 : idx + block_size + 1] for idx in indices], dim=0
    )

    return input.to(device), target.to(device)


def train_and_validate(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_steps: int,
    batch_size: int,
    train_data: Tensor,
    val_data: Tensor,
    block_size: int,
    rank: int | torch.device,
    use_amp: bool,
    lr_scheduler: Optional[LRScheduler] = None,
    log_freq_loss: int = 100,
    max_norm: Optional[float] = None,
    wandb_logging: bool = False,
) -> Dict[torch.Tensor, torch.Tensor]:
    """
    Train and validate the model.

    Args:
        model: Model to train.
        optimizer: Optimizer to use.
        num_steps: Number of iterations to train.
        batch_size: Batch size.
        train_data: Training data.
        val_data: Validation data.
        block_size: Maximum context length.
        rank: Device on which the code is executed.
        use_amp: Whether to use automatic mixed precision.
        train_loader: Dataloader for the training set.
        val_loader: Dataloader for the validation set.
        lr_scheduler: Learning rate scheduler.
        log_freq_loss: Frequency at which to log the loss.
        max_norm: Maximum norm of the gradients.
        world_size: Number of processes participating in the job. Used to get
            the number of iterations correctly in a DDP setup.
        wandb_logging: API key for Weights & Biases.

    Returns:
        checkpoint: Checkpoint of the model.
    """

    # loss function:
    cce_mean = nn.CrossEntropyLoss(reduction="mean")

    # auxiliary variables:
    train_losses, val_losses = [], []
    min_val_loss = float("inf")

    # for automatic mixed precision (AMP):
    scaler = GradScaler(enabled=use_amp)

    # start timing:
    start_time = start_timer(device=rank)

    for step in range(num_steps):
        model.train()

        X, Y = get_batch(
            data=train_data,
            batch_size=batch_size,
            block_size=block_size,
            device=rank,
        )  # X: `[N, block_size]`, Y: `[N, block_size]`
        mask = get_subsequent_mask(size=X.shape[1], rank=rank)

        optimizer.zero_grad()
        if lr_scheduler is not None:
            lr_scheduler.step()

        with autocast(
            device_type=X.device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            output = model(X, mask=mask)  # `[N, block_size, vocab_size]`
            loss = cce_mean(
                output.reshape(-1, output.shape[-1]), Y.reshape(-1)
            )

        scaler.scale(loss).backward()
        if max_norm is not None:
            scaler.unscale_(optimizer)
            for param_group in optimizer.param_groups:
                clip_grad_norm_(param_group["params"], max_norm=max_norm)
        scaler.step(optimizer)
        scaler.update()

        train_losses.append(loss.item())

        # validation stuff:
        model.eval()
        with torch.no_grad():
            X, Y = get_batch(
                data=val_data,
                batch_size=batch_size,
                block_size=block_size,
                device=rank,
            )
            mask = get_subsequent_mask(size=X.shape[1], rank=rank)

            with autocast(
                device_type=X.device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                val_output = model(X, mask=mask)
                val_loss = cce_mean(
                    val_output.reshape(-1, val_output.shape[-1]),
                    Y.reshape(-1),
                )
            val_losses.append(val_loss.item())

        # log train and val losses:
        if step % log_freq_loss == 0 and rank in [0, torch.device("cpu")]:
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            # reset losses:
            train_losses, val_losses = [], []

            logging.info(
                f"step {step}: train loss: {train_loss:.4f}, val loss: "
                f"{val_loss:.4f}, lr: {optimizer.param_groups[0]['lr']}"
            )
            # log to Weights & Biases
            if wandb_logging:
                wandb.log(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "step": step,
                    },
                    step=step,
                )

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                checkpoint = {
                    "state_dict": deepcopy(model.state_dict()),
                    "optimizer": deepcopy(optimizer.state_dict()),
                    "val_loss": val_loss,
                    "step": step,
                }

    if rank in [0, torch.device("cpu")]:
        end_timer_and_log(
            start_time=start_time,
            device=rank,
            local_msg=f"Training {num_steps} steps",
        )

    return checkpoint


def start_timer(device: torch.device | int) -> float:
    """
    Start the timer.

    Args:
        device: Device on which the code is executed. Can also be an int
            representing the GPU ID.

    Returns:
        Time at which the training started.
    """
    gc.collect()

    # check if device is a ``torch.device`` object; if not, assume it's an
    # int and convert it
    if not isinstance(device, torch.device):
        device = torch.device(
            f"cuda:{device}" if isinstance(device, int) else "cpu"
        )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.synchronize()

    return perf_counter()


def end_timer_and_log(
    start_time: float, device: torch.device | int, local_msg: str = ""
) -> float:
    """
    End the timer and print the time it took to execute the code as well as the
    maximum memory used by tensors.

    Args:
        start_time: Time at which the training started.
        device: Device on which the code was executed. Can also be an int
            representing the GPU ID.
        local_msg: Local message to print.

    Returns:
        Time it took to execute the code.
    """

    # check if device is a ``torch.device`` object; if not, assume it's an
    # int and convert it
    if not isinstance(device, torch.device):
        device = torch.device(
            f"cuda:{device}" if isinstance(device, int) else "cpu"
        )

    if device.type == "cuda":
        torch.cuda.synchronize()

    time_diff = perf_counter() - start_time

    msg = f"{local_msg}\n\tTotal execution time = {time_diff:.3f} [sec]"
    if device.type == "cuda":
        msg += (
            f"\n\tMax memory used by tensors = "
            f"{torch.cuda.max_memory_allocated(device=device) / 1024**2:.3f} "
            "[MB]"
        )
    logging.info(msg)

    return time_diff


def load_checkpoint(
    model: nn.Module,
    checkpoint: dict[torch.Tensor, torch.Tensor],
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> None:
    """Load an existing checkpoint of the model to continue training.

    Args:
        model: NN for which state dict is loaded.
        checkpoint: Checkpoint dictionary.
        optimizer: Optimizer for which state dict is loaded.
    """
    model.load_state_dict(state_dict=checkpoint["state_dict"])
    loading_msg = "=> Checkpoint loaded."

    if optimizer is not None:
        optimizer.load_state_dict(state_dict=checkpoint["optimizer"])

    if "epoch" in checkpoint.keys():
        loading_msg += f" It had been saved at epoch {checkpoint['epoch']}."
    elif "step" in checkpoint.keys():
        loading_msg += f" It had been saved at step {checkpoint['step']}."

    if "val_loss" in checkpoint.keys():
        loading_msg += f" Validation loss: {checkpoint['val_loss']:.4f}."

    if "val_acc" in checkpoint.keys():
        loading_msg += (
            f" Validation accuracy: {100 * checkpoint['val_acc']:.2f} %."
        )
    logging.info(loading_msg)


def save_checkpoint(state: Dict, filename: str = "my_checkpoint.pt") -> None:
    """Creates a model checkpoint to save and load a model.

    Params:
        state: State of model and optimizer in a dictionary.
        filename: The name of the checkpoint.
    """
    log_msg = f"\n=> Saving checkpoint '{filename}' "
    if "val_loss" in state.keys():
        log_msg += (
            f"corresponding to a validation loss of {state['val_loss']:.4f} "
        )
    if "val_acc" in state.keys():
        log_msg += (
            f"and a validation accuracy of {100 * state['val_acc']:.2f} % "
        )
    if "epoch" in state.keys():
        log_msg += f"at epoch {state['epoch']}."
    logging.info(log_msg)

    torch.save(state, filename)


def log_parameter_table(model: nn.Module) -> None:
    """Log the number of parameters per module.

    Args:
        model: Model for which we want the total number of parameters.
    """
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    logging.info(f"{table}\nTotal trainable params: {total_params}")


@torch.no_grad()
def generate_text(
    model: nn.Module,
    max_new_tokens: int,
    block_size: int,
    use_amp: bool,
    vocab: Optional[List[str]],
    rank: str | int | torch.device,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> None:
    """
    Generate text from the model.

    Args:
        model: Transformer.
        max_new_tokens: Maximum number of tokens to generate.
        block_size: Maximum context length for predictions.
        use_amp: Whether to use automatic mixed precision.
        vocab: Vocabulary. If not provided, GPT2 encoding is used.
        rank: Device on which the code is executed.
        temperature: Temperature for sampling. For `temperature > 1`,
            predictions will be more diverse, for `temperature < 1`,
            predictions will be more conservative.
        top_k: Top-k sampling.
    """
    model.eval()

    if vocab is not None:
        start_ids = encode("\n", vocab=vocab)
    else:
        # assume GPT-2 encoding
        tokenizer = tiktoken.get_encoding("gpt2")
        start_ids = tokenizer.encode(
            text="\n", allowed_special={"<|endoftext|>"}
        )
    x = torch.tensor(start_ids, dtype=torch.long, device=rank).unsqueeze(dim=0)

    with autocast(
        device_type=x.device.type,
        dtype=torch.float16,
        enabled=use_amp,
    ):
        gen_tok = model.generate(
            x,
            max_new_tokens=max_new_tokens,
            block_size=block_size,
            temperature=temperature,
            top_k=top_k,
        )

    if vocab is not None:
        generated_text = decode(gen_tok.squeeze(dim=0).tolist(), vocab=vocab)
    else:
        generated_text = tokenizer.decode(
            tokens=gen_tok.squeeze(dim=0).tolis()
        )
    logging.info(f"Generated text:\n\n{generated_text}")
