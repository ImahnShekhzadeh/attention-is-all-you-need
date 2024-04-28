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

import numpy as np
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


def get_dataset(
    train_split: float = 0.8,
) -> Tuple[Tensor, Tensor, List[str], int]:
    """
    Get Tiny Shakespeare dataset.

    Args:
        train_split: Fraction of the dataset to use for training.

    Returns:
        Text dataset, vocabulary and its size.
    """

    assert (
        0 < train_split < 1
    ), f"Train split should be > 0 and < 1, but is instead {train_split}"

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    # https://stackoverflow.com/a/7244263
    with urllib.request.urlopen(url) as response, open(
        "input.txt", "wb"
    ) as out_file:
        shutil.copyfileobj(response, out_file)

    with open("input.txt", "r") as f:
        text = f.read()
    os.remove("input.txt")

    logging.info(f"Length of dataset in characters: {len(text)}")
    logging.info(f"First 100 characters of dataset: {text[:100]}")

    # get vocabulary and its size
    vocab = sorted(list(set(text)))
    vocab_size = len(vocab)
    logging.info(
        f"Vocabulary size: {vocab_size}\nVocabulary: {' '.join(vocab)}"
    )

    # tokenize the text
    data = torch.tensor(encode(text, vocab=vocab), dtype=torch.long)
    logging.info(
        f"Encoded text: {data}\nShape: {data.shape}, dtype: {data.dtype}"
    )

    # make train-val split
    num_train_examples = int(train_split * len(data))
    train_data = data[:num_train_examples]
    val_data = data[num_train_examples:]

    return train_data, val_data, vocab, vocab_size


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


def get_batch(
    data: Tensor, batch_size: int, block_size: int
) -> Tuple[Tensor, Tensor]:
    """
    Get batch of data.

    Args:
        data: Train or validation data.
        batch_size: int

    Returns:
        Batch.
    """
    pass


def get_subsequent_mask(size: int, rank: int | torch.device) -> torch.Tensor:
    """
    Define mask to prevent the decoder from attending to subsequent tokens,
    also cf. https://peterbloem.nl/blog/transformers.

    Args:
        size: Size of the square mask.
        rank: Device.

    Returns:
        Subsequent mask, shape: `(size, size)`.
    """

    mask = torch.triu(
        torch.ones(
            size,
            size,
            device=rank,
        ),
        diagonal=1,
    )

    return mask


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
    tgt_mask: Optional[torch.Tensor] = None,
) -> Dict[torch.Tensor, torch.Tensor]:
    """
    Train and validate the model. For the memory key padding mask of the
    transformer, the padding mask of the source sequence is taken, as done
    in [1].

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
        tgt_mask: Look-ahead mask for the decoder, of shape
            `(seq_length, seq_length)`, to prevent the decoder from attending
             to subsequent tokens in the sequence.

    Returns:
        checkpoint: Checkpoint of the model.

    [1] https://pytorch.org/tutorials/beginner/translation_transformer.html
    """

    # loss function:
    cce_mean = nn.CrossEntropyLoss(reduction="mean", ignore_index=pad_token_id)

    start_time = start_timer(device=rank)
    train_losses, val_losses = [], []
    min_val_loss = float("inf")

    scaler = GradScaler(enabled=use_amp)

    for epoch in range(num_epochs):
        t0 = start_timer(device=rank)
        trainingLoss_perEpoch, valLoss_perEpoch = [], []

        for batch_idx, dict in enumerate(train_loader):
            model.train()

            src_tokens = dict["source"].to(rank)  # `(N, S)`
            src_key_padding_mask = src_tokens == pad_token_id  # `(N, S)`
            target_tokens = torch.cat(
                (
                    start_token_id
                    * torch.ones(
                        (src_tokens.shape[0], 1), dtype=src_tokens.dtype
                    ),
                    dict["target"],
                ),
                dim=1,
            ).to(
                rank
            )  # `(N, T + 1)`
            decoder_tokens = target_tokens[
                :, :-1
            ]  # input to decoder, `(N, T)`
            labels = target_tokens[:, 1:]  # `(N, T)`
            tgt_key_padding_mask = decoder_tokens == pad_token_id  # `(N, T)`

            optimizer.zero_grad()
            if lr_scheduler is not None:
                lr_scheduler.step()

            with autocast(
                device_type=src_tokens.device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                # `[N, seq_length, vocab_size]`
                output = model(
                    src_tokens,
                    decoder_tokens,
                    tgt_mask=tgt_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=src_key_padding_mask,
                )
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
                src_key_padding_mask = src_tokens == pad_token_id
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
                ]  # input to decoder, `(N, T)`
                tgt_key_padding_mask = (
                    decoder_tokens == pad_token_id
                )  # `(N, T)`
                labels = target_tokens[:, 1:]  # `(N, T)`

                with autocast(
                    device_type=src_tokens.device.type,
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    val_output = model(
                        src_tokens,
                        decoder_tokens,
                        tgt_mask=tgt_mask,
                        src_key_padding_mask=src_key_padding_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        memory_key_padding_mask=src_key_padding_mask,
                    )
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
                batch_size = val_output.shape[0]

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
                "val_loss": val_losses[epoch],
                "epoch": epoch,
            }

        if rank in [0, torch.device("cpu")]:
            # log to Weights & Biases
            if wandb_logging:
                wandb.log(
                    {
                        "train_loss": train_losses[epoch],
                        "val_loss": val_losses[epoch],
                        "epoch": epoch,
                    },
                    step=epoch,
                )

            logging.info(
                f"\nEpoch {epoch}: {perf_counter() - t0:.3f} [sec]\t"
                f"Mean train/val loss: {train_losses[epoch]:.4f}/"
                f"{val_losses[epoch]:.4f}\n"
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
    use_amp: bool,
    test_loader: DataLoader,
    start_token_id: int,
    pad_token_id: int,
    rank: int | torch.device,
) -> List[List[str]]:
    """
    Generate text from the model.

    Args:
        model: Transformer.
        tokenizer: Tokenizer.
        use_amp: Whether to use automatic mixed precision.
        test_loader: Dataloader for the test set.
        start_token_id: ID of the start token.
        rank: Device on which the code is executed.

    Returns:
        Generated text.
    """
    model.eval()
    generated_ids = []

    for test_dict in test_loader:
        src_tokens = test_dict["source"].to(rank)  # `(N, S)`
        src_key_padding_mask = src_tokens == pad_token_id  # `(N, S)`

        decoder_tokens = start_token_id * torch.ones(
            (src_tokens.shape[0], 1), dtype=src_tokens.dtype
        ).to(rank)

        while decoder_tokens.shape[1] < src_tokens.shape[1]:
            with autocast(
                device_type=src_tokens.device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                tgt_mask = get_subsequent_mask(
                    size=decoder_tokens.shape[1], rank=rank
                )

                output = model(
                    src_tokens,
                    decoder_tokens,
                    tgt_mask=tgt_mask,
                    src_key_padding_mask=src_key_padding_mask,
                )  # `(N, decoder_tokens.shape[1], vocab_size)`

            generated_tokens = output.argmax(dim=2)[:, -1].unsqueeze(dim=1)

            # append the generated token IDs to decoder tokens
            decoder_tokens = torch.cat(
                (decoder_tokens, generated_tokens), dim=1
            )

        generated_ids.extend(decoder_tokens[:, 1:].cpu().tolist())

    generated_text = tokenizer.decode_batch(generated_ids)  # `List[str]`
    generated_text = [[item] for item in generated_text]  # `List[List[str]]`
    logging.info(f"Generated translations:\n\n{generated_text}")

    return generated_text


def compute__bleu_score(test_data: List[Dict], generated_data: List) -> Dict:
    """
    Compute the BLEU score of the model.

    Args:
        test_data: List containing test data (strings) for both the source and
            target languages.
        generated_data: Translated sentences.

    Returns:
        Dictionary containing BLEU score, precisions, brevity penalty, length
        ratio, etc.
    """
    reference_data = []
    for dict in test_data:
        reference_data.append([dict["en"]])

    bleu = evaluate.load("bleu")
    results = bleu.compute(
        predictions=generated_data,
        references=reference_data,
    )
    return results
