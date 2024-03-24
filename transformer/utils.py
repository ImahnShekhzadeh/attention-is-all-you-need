"""Utility functions and classes."""
import gc
import json
import logging
import os
import shutil
import sys
from argparse import ArgumentParser, Namespace
from copy import deepcopy
from datetime import datetime as dt
from math import ceil
from time import perf_counter
from typing import Dict, Iterator, List, Optional, Tuple, Union

import datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from prettytable import PrettyTable
from scheduler import LRScheduler
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from torch import Tensor, autocast
from torch import distributed as dist
from torch import multiprocessing as mp
from torch import nn
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, DistributedSampler, IterableDataset
from torchtext.data.metrics import bleu_score


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

    # create saving dir if non-existent, check if saving path is empty, and
    # copy JSON config file there
    os.makedirs(args.saving_path, exist_ok=True)
    if not len(os.listdir(args.saving_path)) == 0:
        raise ValueError(
            f"Saving path `{args.saving_path}` is not empty! Please provide "
            "another path."
        )
    shutil.copy(src=args.config, dst=args.saving_path)

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
            args.num_epochs > 0
        ), "Number of epochs should be greater than 0 when training."
    else:
        if args.num_epochs > 0:
            logging.warning(
                "`--train` flag not set, but `--num_epochs > 0` is provided."
                "Training will NOT be performed."
            )
        assert args.loading_path is not None, (
            "`--train` flag was not passed, so please provide a path to load "
            "the model from."
        )


def get_bpe_tokenizer(
    seq_length: int,
    tokenizer_file: Optional[str] = None,
    iterator: Union[Iterator, List[str]] = None,
    vocab_size: Optional[int] = None,
    min_frequency: Optional[int] = None,
) -> Tokenizer:
    """
    Get a tokenizer with byte-pair encoding (BPE) according to the
    instructions in [1]. If a tokenizer path is provided, the tokenizer will be
    loaded from this path. If no tokenizer is provided, the tokenizer will be
    trained on the provided files.

    Args:
        seq_length: Sequence length; if sentence contains less tokens than
            `seq_length`, it will be padded, otherwise truncated.
        iterator: Any iterator over strings or list of strings with which the
            tokenizer is trained [2].
        files: List of files to train the tokenizer on.
        vocab_size: Vocabulary size.
        min_frequency: Minimum frequency a pair must have to produce a merge
            operation.

    [1] https://huggingface.co/docs/tokenizers/quicktour
    [2] https://github.com/huggingface/tokenizers/blob/c893204c45d7f2cd66958731dd7779548ca54ad5/bindings/python/py_src/tokenizers/__init__.pyi#L1089
    """
    if tokenizer_file is None:
        assert (
            iterator is not None
            and vocab_size is not None
            and min_frequency is not None
        ), (
            "If no tokenizer is provided, the following arguments must be "
            "provided: `iterator`, `vocab_size`, `min_frequency`."
        )

        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=["[SOS]", "[UNK]", "[PAD]"],
        )

        tokenizer.pre_tokenizer = Whitespace()

        tokenizer.train_from_iterator(iterator, trainer)

        tokenizer.save(f"bpe_tokenizer_{vocab_size // 1000}k.json")
    else:
        assert os.path.exists(
            tokenizer_file
        ), "Please provide a valid path to the tokenizer file."
        assert tokenizer_file.endswith(
            ".json"
        ), "If a tokenizer is provided, it must be a JSON file."

        tokenizer = Tokenizer.from_file(tokenizer_file)
        logging.info(
            f"=> Tokenizer `{tokenizer_file}` with a vocabulary size of "
            f"{tokenizer.get_vocab_size()} loaded."
        )

    # enable padding and truncation.
    pad_token = "[PAD]"
    tokenizer.enable_padding(
        pad_id=tokenizer.token_to_id(pad_token),
        pad_token=pad_token,
        length=seq_length,
        direction="right",
    )
    tokenizer.enable_truncation(
        max_length=seq_length,
    )

    return tokenizer


def get_datasets_and_tokenizer(
    data: datasets.dataset_dict.DatasetDict,
    seq_length: int,
    tokenizer_file: str = "bpe_tokenizer_30k.json",
    vocab_size: Optional[int] = None,
    min_frequency: Optional[int] = None,
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor], Tokenizer]:
    """
    Get the train, val and test datasets of the IWSLT 2017 DE-EN dataset.

    Args:
        data: Datasets (train, val, test).
        seq_length: Sequence length; if sentence contains less tokens than
            `seq_length`, it will be padded, otherwise truncated.
        tokenizer_file: Path to the tokenizer file.
        vocab_size: Vocabulary size.
        min_frequency: Minimum frequency a pair must have to produce a merge
            operation.

    Returns:
        Dictionary containing the tokenized texts for the train, val and test
            split for both German and English and the tokenizer used for the
            encoding.
    """

    # load tokenizer from file path or train from scratch
    tokenizer_args = {
        "seq_length": seq_length,
    }

    if tokenizer_file is None:
        all_sentences = []

        for split in data.keys():
            for i in range(len(data[split])):
                all_sentences.extend(data[split]["translation"][i].values())

        # save list to numpy
        np.save("transformer/all_sentences.npy", all_sentences)

        tokenizer = get_bpe_tokenizer(
            **tokenizer_args
            | {
                "vocab_size": vocab_size,
                "min_frequency": min_frequency,
                "iterator": all_sentences,
            },
        )
    else:
        tokenizer = get_bpe_tokenizer(
            **tokenizer_args | {"tokenizer_file": tokenizer_file}
        )

    def tokenize_text(batch: List[Dict]) -> Dict[str, Tensor]:
        """
        Tokenize test corresponding to train, validation or test splits.
        This function is specifically written for the IWSLT 2017 dataset.

        Args:
            batch: List of dictionaries containing the text to tokenize.
                Dictionary contains keys "de" and "en" for German and English
                translations respectively.

        Returns:
            Dictionary containing the tokenized text for both German and
            English translations.
        """

        src_ids, target_ids = [], []

        for dict in batch:
            src_ids.append(tokenizer.encode(dict["de"]).ids)
            target_ids.append(tokenizer.encode(dict["en"]).ids)

        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
        }

    train__dict_ids = tokenize_text(data["train"]["translation"])
    val__dict_ids = tokenize_text(data["validation"]["translation"])
    test__dict_ids = tokenize_text(data["test"]["translation"])

    return train__dict_ids, val__dict_ids, test__dict_ids, tokenizer


def get_dataloaders(
    train_dataset: IterableDataset,
    val_dataset: IterableDataset,
    test_dataset: IterableDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    use_ddp: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get the dataloaders for the train, validation and test set.

    Args:
        train_dataset: Training set.
        val_dataset: Validation set.
        test_dataset: Test set.
        batch_size: Batch size.
        num_workers: Number of subprocesses used in the dataloaders.
        pin_memory: Whether tensors are copied into CUDA pinned memory.
        use_ddp: Whether to use DDP.

    Returns:
        Train, val and test loader
    """

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False if use_ddp else True,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    train_loader = DataLoader(
        dataset=train_dataset,
        sampler=DistributedSampler(train_dataset) if use_ddp else None,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        sampler=DistributedSampler(val_dataset) if use_ddp else None,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        **loader_kwargs,
    )

    return train_loader, val_loader, test_loader


def train_and_validate(
    pad_token_id: int,
    start_token_id: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    rank: int | torch.device,
    use_amp: bool,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr_scheduler: Optional[LRScheduler] = None,
    freq_output__train: Optional[int] = 10,
    freq_output__val: Optional[int] = 10,
    max_norm: Optional[float] = None,
    world_size: Optional[int] = None,
    wandb_logging: bool = False,
) -> Dict[torch.Tensor, torch.Tensor]:
    """
    Train and validate the model.

    Args:
        pad_token_id: ID of the pad token `[PAD]`.
        start_token_id: ID of the start token `[SOS]`.
        model: Model to train.
        optimizer: Optimizer to use.
        num_epochs: Number of epochs to train the model.
        rank: Device on which the code is executed.
        use_amp: Whether to use automatic mixed precision.
        train_loader: Dataloader for the training set.
        val_loader: Dataloader for the validation set.
        lr_scheduler: Learning rate scheduler.
        freq_output__train: Frequency at which to print the training info.
        freq_output__val: Frequency at which to print the validation info.
        max_norm: Maximum norm of the gradients.
        world_size: Number of processes participating in the job. Used to get
            the number of iterations correctly in a DDP setup.
        wandb_logging: API key for Weights & Biases.

    Returns:
        checkpoint: Checkpoint of the model.
    """

    # loss function:
    cce_mean = nn.CrossEntropyLoss(reduction="mean", ignore_index=pad_token_id)

    start_time = start_timer(device=rank)
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    min_val_loss = float("inf")

    scaler = GradScaler(enabled=use_amp)

    for epoch in range(num_epochs):
        t0 = start_timer(device=rank)
        trainingLoss_perEpoch, valLoss_perEpoch = [], []
        num_correct, num_samples, val_num_correct, val_num_samples = 0, 0, 0, 0

        for batch_idx, dict in enumerate(train_loader):
            model.train()

            # tokens in source language `[N, seq_length]`
            src_tokens = dict["source"].to(rank)
            # tokens in target language, `[N, seq_length + 1]`
            target_tokens = torch.cat(
                (
                    start_token_id
                    * torch.ones(
                        (src_tokens.shape[0], 1), dtype=src_tokens.dtype
                    ),
                    dict["target"],
                ),
                dim=1,
            ).to(rank)
            decoder_tokens = target_tokens[
                :, :-1
            ]  # input to decoder, `[N, seq_length]`
            labels = target_tokens[:, 1:]  # `[N, seq_length]`

            optimizer.zero_grad()
            if lr_scheduler is not None:
                lr_scheduler.step()

            with autocast(
                device_type=src_tokens.device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                # `[N, seq_length, vocab_size]`
                output = model(src_tokens, decoder_tokens)
                loss = cce_mean(
                    # `[N * seq_length, vocab_size]`
                    output.reshape(-1, output.shape[-1]),
                    # `[N * seq_length]`
                    labels.reshape(-1),
                )

            scaler.scale(loss).backward()
            if max_norm is not None:
                scaler.unscale_(optimizer)
                for param_group in optimizer.param_groups:
                    clip_grad_norm_(param_group["params"], max_norm=max_norm)
            scaler.step(optimizer)
            scaler.update()

            trainingLoss_perEpoch.append(
                loss.cpu().item() * src_tokens.shape[0]
            )

            with torch.no_grad():
                # ignore padding tokens
                labels = torch.where(labels == pad_token_id, -1, labels)
                # calculate accuracy
                _, max_indices = output.max(dim=2, keepdim=False)
                num_correct += (max_indices == labels).sum().cpu().item()
                num_samples += output.shape[0] * output.shape[1]

            if rank in [0, torch.device("cpu")]:
                log__batch_info(
                    batch_idx=batch_idx,
                    loader=train_loader,
                    epoch=epoch,
                    loss=loss,
                    mode="train",
                    frequency=freq_output__train,
                )

        # validation stuff:
        model.eval()
        with torch.no_grad():
            for val_batch_idx, val_dict in enumerate(val_loader):
                src_tokens = val_dict["source"].to(rank)
                target_tokens = torch.cat(
                    (
                        start_token_id
                        * torch.ones(
                            (src_tokens.shape[0], 1), dtype=src_tokens.dtype
                        ),
                        val_dict["target"],
                    ),
                    dim=1,
                ).to(rank)
                decoder_tokens = target_tokens[
                    :, :-1
                ]  # input to decoder, `[N, seq_length]`
                labels = target_tokens[:, 1:]  # `[N, seq_length]`

                with autocast(
                    device_type=src_tokens.device.type,
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    val_output = model(src_tokens, decoder_tokens)
                    val_loss = (
                        cce_mean(
                            # `[N * seq_length, vocab_size]`
                            val_output.reshape(-1, val_output.shape[-1]),
                            # `[N * seq_length]`
                            labels.reshape(-1),
                        )
                        .cpu()
                        .item()
                        * src_tokens.shape[0]
                    )

                valLoss_perEpoch.append(val_loss)

                # ignore padding tokens
                labels = torch.where(labels == pad_token_id, -1, labels)
                # calculate accuracy
                _, val_max_indices = val_output.max(dim=2, keepdim=False)
                val_num_correct += (
                    (val_max_indices == labels).sum().cpu().item()
                )
                batch_size = val_output.shape[0]
                val_num_samples += batch_size * val_output.shape[1]

                if rank in [0, torch.device("cpu")]:
                    log__batch_info(
                        batch_idx=val_batch_idx,
                        loader=val_loader,
                        epoch=epoch,
                        loss=val_loss / batch_size,
                        mode="val",
                        frequency=freq_output__val,
                    )

        train_losses.append(
            np.sum(trainingLoss_perEpoch, axis=0) / len(train_loader.dataset)
        )
        val_losses.append(
            np.sum(valLoss_perEpoch, axis=0) / len(val_loader.dataset)
        )
        if val_losses[epoch] < min_val_loss:
            min_val_loss = val_losses[epoch]
            checkpoint = {
                "state_dict": deepcopy(model.state_dict()),
                "optimizer": deepcopy(optimizer.state_dict()),
            }

        # Calculate accuracies for each epoch:
        train_accs.append(num_correct / num_samples)
        val_accs.append(val_num_correct / val_num_samples)

        if rank in [0, torch.device("cpu")]:
            # log to Weights & Biases
            if wandb_logging:
                wandb.log(
                    {
                        "train_loss": train_losses[epoch],
                        "val_loss": val_losses[epoch],
                        "train_acc": train_accs[epoch],
                        "val_acc": val_accs[epoch],
                        "epoch": epoch,
                    },
                    step=epoch,
                )

            logging.info(
                f"\nEpoch {epoch}: {perf_counter() - t0:.3f} [sec]\t"
                f"Mean train/val loss: {train_losses[epoch]:.4f}/"
                f"{val_losses[epoch]:.4f}\tTrain/val acc: "
                f"{1e2 * train_accs[epoch]:.2f} %/{1e2 * val_accs[epoch]:.2f} %\n"
            )
        model.train()

    if world_size is not None:
        num_iters = (
            ceil(
                len(train_loader.dataset)
                / (world_size * train_loader.batch_size)
            )
            * num_epochs
        )
    else:
        num_iters = (
            ceil(len(train_loader.dataset) / train_loader.batch_size)
            * num_epochs
        )

    if rank in [0, torch.device("cpu")]:
        end_timer_and_log(
            start_time=start_time,
            device=rank,
            local_msg=(
                f"Training {num_epochs} epochs ({num_iters} iterations)"
            ),
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


def format_line(
    mode: str,
    epoch: int,
    current_samples: int,
    total_samples: int,
    percentage: float,
    loss: Tensor,
) -> None:
    assert mode.lower() in ["train", "val"]

    # calculate maximum width for each part
    max_epoch_width = len(f"{mode.capitalize()} epoch: {epoch}")
    max_sample_info_width = len(f"[{total_samples} / {total_samples} (100 %)]")

    # format each part
    epoch_str = f"{mode.capitalize()} epoch: {epoch}".ljust(max_epoch_width)
    padded__current_sample = str(current_samples).zfill(
        len(str(total_samples))
    )
    sample_info_str = f"[{padded__current_sample} / {total_samples} ({percentage:06.2f} %)]".ljust(
        max_sample_info_width
    )
    loss_str = f"Loss: {loss:.4f}"

    return f"{epoch_str}  {sample_info_str}  {loss_str}"


def log__batch_info(
    mode: str,
    batch_idx: int,
    loader: DataLoader,
    epoch: int,
    loss: Tensor,
    frequency: int = 1,
) -> None:
    """
    Print the current batch information.

    Params:
        mode: Mode in which the model is in. Either "train" or "val".
        batch_idx: Batch index.
        loader: Train or validation Dataloader.
        epoch: Current epoch.
        loss: Loss of the current batch.
        frequency: Frequency at which to print the batch info.
    """
    assert mode.lower() in ["train", "val"]
    assert type(frequency) == int

    if batch_idx % frequency == 0:
        if batch_idx == len(loader) - 1:
            current_samples = len(loader.dataset)
        else:
            current_samples = (batch_idx + 1) * loader.batch_size

        total_samples = len(loader.dataset)
        prog_perc = 100 * current_samples / total_samples

        formatted_line = format_line(
            mode=mode,
            epoch=epoch,
            current_samples=current_samples,
            total_samples=total_samples,
            percentage=prog_perc,
            loss=loss,
        )
        logging.info(f"{formatted_line}")


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
    if optimizer is not None:
        optimizer.load_state_dict(state_dict=checkpoint["optimizer"])
    logging.info("=> Checkpoint loaded.")


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """Creates a model checkpoint to save and load a model.

    Params:
        state (dictionary)      -- The state of the model and optimizer in a
            dictionary.
        filename (pth.tar)      -- The name of the checkpoint.
    """
    torch.save(state, filename)
    logging.info("\n=> Saving checkpoint")


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


def check_accuracy(loader, model, mode, device, pad_token_id):
    """
    Check the accuracy of a given model on a given dataset.

    Params:
        loader (torch.utils.data.DataLoader)        -- The dataloader of the
            dataset on which we want to check the accuracy.
        model (torch.nn)                            -- Model for which we want
            the total number of parameters.
        mode (str):                                 -- Mode in which the model
            is in. Either "train" or "test".
        device (torch.device)                       -- Device on which the code
            was executed.
        pad_token_id: ID of the pad token.
    """
    assert mode in ["train", "test"]

    model.eval()
    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for dict in loader:
            tokens = dict["source"].to(device)  # `[N, seq_length]`
            labels = dict["target"].to(device)  # `[N, seq_length]`
            output = model(tokens, labels)
            # ignore padding tokens
            labels = torch.where(labels == pad_token_id, -1, labels)
            _, max_indices = output.max(dim=2, keepdim=False)
            num_correct += (max_indices == labels).sum().cpu().item()
            num_samples += output.shape[0] * output.shape[1]

        logging.info(
            f"{mode.capitalize()} data: Got {num_correct}/{num_samples} with "
            f"accuracy {(100 * num_correct / num_samples):.2f} %"
        )


# TODO: expand this function with while-loop to iteratively feed
# the model with the generated tokens
@torch.no_grad()
def generate_text(
    model: nn.Module,
    tokenizer: Tokenizer,
    use_amp: bool,
    test_loader: DataLoader,
    start_token_id: int,
    rank: int | torch.device,
) -> torch.Tensor:
    """
    Generate text from the model.

    Args:
        model: Transformer.
        tokenizer: Tokenizer.
        use_amp: Whether to use automatic mixed precision.
        test_loader: Dataloader for the test set.
        start_token_id: ID of the start token.
        device: Device on which the code is executed.

    Returns:
        Generated text.
    """
    model.eval()
    generated_ids = []

    for test_dict in test_loader:
        # tokens in source language `[N, seq_length]`
        src_tokens = test_dict["source"].to(rank)

        decoder_tokens = start_token_id * torch.ones(
            (src_tokens.shape[0], 1), dtype=src_tokens.dtype
        ).to(rank)

        with autocast(
            device_type=src_tokens.device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            # `[N, XXX, vocab_size]`
            output = model(src_tokens, decoder_tokens)

        # append the generated token IDs to decoder tokens
        decoder_tokens = torch.cat(
            (decoder_tokens, output.argmax(dim=2)), dim=1
        )
        generated_ids.extend(output.argmax(dim=2).cpu().tolist())

    generated_text = tokenizer.decode_batch(generated_ids)
    logging.info(f"Generated translations:\n\n{generated_text}")

    return generated_text


def compute__bleu_score(
    test_data: List[Dict],
    max__n_gram: int,
    generated__token_ids: torch.Tensor,
    tokenizer: Tokenizer,
) -> float:
    """
    Compute the BLEU score of the model.

    Args:
        test_data: List containing test data (strings) for both the source and
            target languages.
        max__n_gram: Maximum n-gram used when calculating BLEU score.
        generated__token_ids: Generated data containing token IDs.
        tokenizer: Tokenizer.

    Returns:
        BLEU score (between 0 and 1).
    """
    reference_data = []
    for dict in test_data:
        reference_data.append(dict["en"])

    generated_data = tokenizer.decode_batch(generated__token_ids.tolist())

    return bleu_score(generated_data, test_data, max__n_gram)
