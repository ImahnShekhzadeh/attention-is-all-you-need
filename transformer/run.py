"""
Run transformer.
"""
import logging
import os
import sys
from argparse import Namespace
from datetime import datetime as dt

import torch
import wandb
from scheduler import LRScheduler
from torch import multiprocessing as mp
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import (
    cleanup,
    generate_text,
    get_dataset,
    load_checkpoint,
    log_parameter_table,
    retrieve_args,
    save_checkpoint,
    setup,
    train_and_validate,
)

from architecture.models import Transformer
from options import get_parser


def main(
    rank: int | torch.device,
    world_size: int,
    args: Namespace,
) -> None:
    """
    Main function.

    Args:
        rank: rank of the current process
        world_size: number of processes
        args: command line arguments
    """

    if args.seed_number is not None:
        torch.manual_seed(args.seed_number)

    if args.use_ddp:
        setup(
            rank=rank,
            world_size=world_size,
        )

    # get dataset
    train_data, val_data, vocab, vocab_size = get_dataset()

    # define transformer
    model = Transformer(
        num__decoder_layers=args.num__decoder_layers,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        vocab_size=vocab_size,
        dim_feedfwd=args.dim_feedfwd,
        dropout_rate=args.dropout_rate,
    )
    model.to(rank)
    if args.use_ddp:
        model = DDP(model, device_ids=[rank])

    # setup Weights & Biases, print # data and log parameter table
    if rank in [0, torch.device("cpu")]:
        wandb_logging = args.wandb__api_key is not None and args.train
        if wandb_logging:
            wandb.login(key=args.wandb__api_key)
            wandb.init(
                project="transformer",
                name=dt.now().strftime("%dp%mp%Y_%Hp%M"),
                config=args,
            )

        log_parameter_table(model)
    else:
        wandb_logging = False

    # compile model if specified
    if args.compile_mode is not None:
        logging.info(f"\nCompiling model in ``{args.compile_mode}`` mode...\n")
        model = torch.compile(model, mode=args.compile_mode, fullgraph=False)

    # Optimizer:
    optimizer = optim.Adam(
        params=model.parameters(),
        lr=1e-3,  # dummy, will be set by scheduler
        betas=(args.beta_1, args.beta_2),
        eps=args.eps,
    )

    # Set network to train mode:
    model.train()

    if args.loading_path is not None:
        if rank == torch.device("cpu"):
            map_location = {"cuda:0": "cpu"}
        else:
            map_location = {"cuda:0": f"cuda:{rank}"}

        load_checkpoint(
            model=model,
            optimizer=optimizer,
            checkpoint=torch.load(
                args.loading_path, map_location=map_location
            ),
        )

    if args.train:
        lr_scheduler = LRScheduler(
            optimizer=optimizer,
            d_model=args.embedding_dim,
            warmup_steps=args.warmup_steps,
            lr_multiplier=args.lr_multiplier,
        )
        checkpoint = train_and_validate(
            model=model,
            optimizer=optimizer,
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            train_data=train_data,
            val_data=val_data,
            block_size=args.block_size,
            rank=rank,
            use_amp=args.use_amp,
            lr_scheduler=lr_scheduler,
            log_freq_loss=args.log_freq_loss,
            max_norm=args.max_norm,
            wandb_logging=wandb_logging,
        )

        if rank in [0, torch.device("cpu")]:
            # save model and optimizer state dicts
            save_checkpoint(
                state=checkpoint,
                filename=os.path.join(
                    args.saving_path,
                    f"cp_{dt.now().strftime('%dp%mp%Y_%Hp%M')}.pt",
                ),
            )

    # destroy process group if DDP was used (for clean exit)
    if args.use_ddp:
        cleanup()

    if rank in [0, torch.device("cpu")]:
        if wandb_logging:
            wandb.finish()

        if args.train:
            # load checkpoint with lowest validation loss for final evaluation;
            # device does not need to be specified, since the checkpoint will
            # be loaded on the CPU or GPU with ID 0 depending on where the
            # checkpoint was saved
            load_checkpoint(model=model, checkpoint=checkpoint)

        model.eval()

        generate_text(
            model=model,
            vocab=vocab,
            block_size=args.block_size,
            use_amp=args.use_amp,
            max_new_tokens=args.max_new_tokens,
            rank=rank,
            temperature=args.temperature,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    parser = get_parser()
    args = retrieve_args(parser)

    # Setup basic configuration for logging
    log_level = logging.INFO
    logging.basicConfig(
        filename=os.path.join(
            args.saving_path, f"run_{dt.now().strftime('%dp%mp%Y_%Hp%M')}.log"
        ),
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Create `StreamHandler` for stdout and add it to root logger
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    logging.getLogger().addHandler(console_handler)

    if args.config is not None and os.path.exists(args.config):
        logging.info(f"Config file '{args.config}' found and loaded.")
    logging.info(args)

    # define world size (number of GPUs)
    world_size = torch.cuda.device_count()

    if torch.cuda.is_available():
        list_gpus = [torch.cuda.get_device_name(i) for i in range(world_size)]
        logging.info(f"\nGPU(s): {list_gpus}\n")

    if args.use_ddp and world_size > 1:
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs of the current node.
        args.batch_size = int(args.batch_size / world_size)
        mp.spawn(main, args=(world_size, args), nprocs=world_size)
    else:
        args.use_ddp = False
        main(
            rank=0 if world_size >= 1 else torch.device("cpu"),
            world_size=1,
            args=args,
        )
