import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets import load_dataset

from tokenizers import (
    Tokenizer,
    models,
    trainers,
    pre_tokenizers,
    normalizers,
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
)

import sys

sys.path.insert(
    0,
    "/app"
)

from shared.model import (
    DecoderTransformerClassifier
)


LABELS = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}


def seed_everything(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


def build_tokenizer(
    texts,
    vocab_size,
    out_path,
):

    tokenizer = Tokenizer(
        models.BPE(
            unk_token="[UNK]"
        )
    )

    tokenizer.normalizer = (
        normalizers.Sequence(
            [
                normalizers.NFKC()
            ]
        )
    )

    tokenizer.pre_tokenizer = (
        pre_tokenizers.Whitespace()
    )

    trainer = trainers.BpeTrainer(

        vocab_size=vocab_size,

        special_tokens=[
            "[PAD]",
            "[UNK]",
            "[BOS]",
            "[EOS]",
        ],
    )

    tokenizer.train_from_iterator(
        texts,
        trainer=trainer,
    )

    tokenizer.save(
        str(out_path)
    )

    return tokenizer


class EncodedDataset(
    torch.utils.data.Dataset
):

    def __init__(
        self,
        split,
        tokenizer,
        max_len,
    ):

        self.rows = split

        self.tokenizer = tokenizer

        self.max_len = max_len

        self.pad_id = (
            tokenizer.token_to_id(
                "[PAD]"
            )
        )

    def __len__(self):

        return len(self.rows)

    def __getitem__(self, i):

        row = self.rows[i]

        ids = (
            self.tokenizer
            .encode(row["text"])
            .ids
        )

        ids = ids[:self.max_len]

        if not ids:

            ids = [
                self.tokenizer.token_to_id(
                    "[UNK]"
                )
            ]

        attention = [1] * len(ids)

        padding_length = (
            self.max_len - len(ids)
        )

        ids += (
            [self.pad_id]
            * padding_length
        )

        attention += (
            [0]
            * padding_length
        )

        return (

            torch.tensor(
                ids,
                dtype=torch.long,
            ),

            torch.tensor(
                attention,
                dtype=torch.long,
            ),

            torch.tensor(
                row["label"],
                dtype=torch.long,
            ),
        )


def evaluate(
    model,
    loader,
    device,
):

    model.eval()

    ys = []

    predictions = []

    with torch.inference_mode():

        for (
            ids,
            mask,
            labels,
        ) in loader:

            ids = ids.to(device)

            mask = mask.to(device)

            logits = model(
                ids,
                mask,
            )

            predicted = (
                logits
                .argmax(-1)
                .cpu()
                .tolist()
            )

            predictions.extend(
                predicted
            )

            ys.extend(
                labels.tolist()
            )

    accuracy = accuracy_score(
        ys,
        predictions,
    )

    macro_f1 = f1_score(
        ys,
        predictions,
        average="macro",
    )

    return accuracy, macro_f1


def train(args):

    seed_everything(
        args.seed
    )

    output_dir = Path(
        args.artifact_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Training device: {device}"
    )

    # -------------------------------------------------
    # 1. LOAD AG NEWS
    # -------------------------------------------------

    dataset = load_dataset(
        "ag_news"
    )

    train_split = (
        dataset["train"]
        .shuffle(
            seed=args.seed
        )
    )

    test_split = dataset["test"]

    # -------------------------------------------------
    # 2. TOKENIZER
    # -------------------------------------------------

    tokenizer_path = (
        output_dir
        / "tokenizer.json"
    )

    if (
        args.rebuild_tokenizer
        or not tokenizer_path.exists()
    ):

        tokenizer = build_tokenizer(

            (
                row["text"]
                for row in train_split
            ),

            args.vocab_size,

            tokenizer_path,
        )

    else:

        tokenizer = (
            Tokenizer.from_file(
                str(tokenizer_path)
            )
        )

    # -------------------------------------------------
    # 3. DATASETS
    # -------------------------------------------------

    train_dataset = EncodedDataset(
        train_split,
        tokenizer,
        args.max_len,
    )

    test_dataset = EncodedDataset(
        test_split,
        tokenizer,
        args.max_len,
    )

    # -------------------------------------------------
    # 4. DATALOADERS
    # -------------------------------------------------

    train_loader = DataLoader(

        train_dataset,

        batch_size=args.batch_size,

        shuffle=True,

        num_workers=args.workers,

        pin_memory=(
            device == "cuda"
        ),
    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=args.eval_batch_size,

        shuffle=False,

        num_workers=args.workers,

        pin_memory=(
            device == "cuda"
        ),
    )

    # -------------------------------------------------
    # 5. MODEL CONFIG
    # -------------------------------------------------

    config = {

        "vocab_size":
            tokenizer.get_vocab_size(),

        "max_len":
            args.max_len,

        "d_model":
            args.d_model,

        "n_heads":
            args.n_heads,

        "n_layers":
            args.n_layers,

        "d_ff":
            args.d_ff,

        "dropout":
            args.dropout,

        "num_classes":
            4,
    }

    model = (
        DecoderTransformerClassifier(
            **config
        )
        .to(device)
    )

    # -------------------------------------------------
    # 6. OPTIMIZER
    # -------------------------------------------------

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=args.lr,

        weight_decay=args.weight_decay,
    )

    criterion = (
        nn.CrossEntropyLoss()
    )

    scaler = torch.amp.GradScaler(

        "cuda",

        enabled=(
            device == "cuda"
        ),
    )

    # -------------------------------------------------
    # 7. TRAINING
    # -------------------------------------------------

    best_f1 = -1.0

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        model.train()

        total_loss = 0.0

        for (
            ids,
            mask,
            labels,
        ) in train_loader:

            ids = ids.to(device)

            mask = mask.to(device)

            labels = labels.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(

                device_type="cuda",

                dtype=torch.float16,

                enabled=(
                    device == "cuda"
                ),
            ):

                logits = model(
                    ids,
                    mask,
                )

                loss = criterion(
                    logits,
                    labels,
                )

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(

                model.parameters(),

                max_norm=1.0,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            total_loss += (
                float(
                    loss.item()
                )
            )

        # -------------------------------------------------
        # 8. EVALUATION
        # -------------------------------------------------

        accuracy, macro_f1 = evaluate(

            model,

            test_loader,

            device,
        )

        average_loss = (
            total_loss
            / len(train_loader)
        )

        print(

            f"epoch={epoch} "
            f"loss={average_loss:.4f} "
            f"accuracy={accuracy:.4f} "
            f"macro_f1={macro_f1:.4f}"
        )

        # -------------------------------------------------
        # 9. CHECKPOINT
        # -------------------------------------------------

        if macro_f1 > best_f1:

            best_f1 = macro_f1

            torch.save(

                model.state_dict(),

                output_dir
                / "model.pt",
            )

            metadata = {

                "model_version":
                    args.model_version,

                "model_config":
                    config,

                "pad_id":
                    tokenizer.token_to_id(
                        "[PAD]"
                    ),

                "labels":
                    LABELS,

                "metrics": {

                    "accuracy":
                        accuracy,

                    "macro_f1":
                        macro_f1,
                },
            }

            (
                output_dir
                / "metadata.json"
            ).write_text(

                json.dumps(
                    metadata,
                    indent=2,
                )
            )

    print(
        "Training complete"
    )

    print(
        {
            "best_macro_f1": best_f1,
            "device": device,
        }
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--artifact-dir",
        default=os.getenv(
            "ARTIFACT_DIR",
            "/app/artifacts",
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--max-len",
        type=int,
        default=int(
            os.getenv(
                "MAX_LEN",
                128,
            )
        ),
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=int(
            os.getenv(
                "TOKENIZER_VOCAB_SIZE",
                16000,
            )
        ),
    )

    parser.add_argument(
        "--d-model",
        type=int,
        default=int(
            os.getenv(
                "D_MODEL",
                256,
            )
        ),
    )

    parser.add_argument(
        "--n-heads",
        type=int,
        default=int(
            os.getenv(
                "N_HEADS",
                8,
            )
        ),
    )

    parser.add_argument(
        "--n-layers",
        type=int,
        default=int(
            os.getenv(
                "N_LAYERS",
                6,
            )
        ),
    )

    parser.add_argument(
        "--d-ff",
        type=int,
        default=int(
            os.getenv(
                "D_FF",
                1024,
            )
        ),
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=float(
            os.getenv(
                "DROPOUT",
                0.1,
            )
        ),
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--model-version",
        default=os.getenv(
            "MODEL_VERSION",
            "latest",
        ),
    )

    parser.add_argument(
        "--rebuild-tokenizer",
        action="store_true",
    )

    train(
        parser.parse_args()
    )