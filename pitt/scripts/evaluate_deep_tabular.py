#!/usr/bin/env python3
"""Evaluate leakage-safe MLP and reduced FT-Transformer models for one target/fold."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
from typing import Any, Callable, Optional, Sequence

import numpy as np

import evaluate_models_bc as common


SCRIPT_VERSION = "1.1"
REGRESSION_TARGETS = {
    "effectiveness": "master__effective",
    "funny": "master__funny",
    "exciting": "master__exciting",
}
CLASSIFICATION_TARGETS = {
    "sentiment": "master__sentiment",
    "topic": "master__topic",
}
MODELS = ("content_mlp", "gated_mlp", "content_ft", "combined_ft")


class DeepEvaluationError(RuntimeError):
    pass


def atomic_directory(final: Path, writer) -> None:
    if final.exists():
        raise DeepEvaluationError(f"refusing to replace existing output: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{final.name}.", dir=final.parent))
    try:
        writer(staging)
        os.replace(staging, final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str):
    import torch

    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise DeepEvaluationError("CUDA was requested but is unavailable")
    return torch.device(requested)


def load_target(cohort: Path, ids: np.ndarray, target_name: str) -> tuple[np.ndarray, str]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise DeepEvaluationError("pandas is required") from exc
    columns = {**REGRESSION_TARGETS, **CLASSIFICATION_TARGETS}
    if target_name not in columns:
        raise DeepEvaluationError(f"unknown target: {target_name}")
    target_column = columns[target_name]
    frame = pd.read_csv(cohort, usecols=["video_id", target_column], dtype={"video_id": str})
    if frame["video_id"].isna().any() or frame["video_id"].duplicated().any():
        raise DeepEvaluationError("cohort target IDs must be complete and unique")
    indexed = frame.set_index("video_id")
    if set(indexed.index) != set(ids):
        raise DeepEvaluationError("cohort target IDs do not match the cache")
    raw = indexed.loc[ids, target_column].to_numpy(dtype=np.float64)
    if not np.isfinite(raw).all():
        raise DeepEvaluationError(f"{target_column} contains non-finite values")
    if target_name in CLASSIFICATION_TARGETS:
        if np.any(raw != np.floor(raw)):
            raise DeepEvaluationError(f"{target_column} must contain integer class IDs")
        raw = raw.astype(np.int64)
    return raw, target_column


def classification_metrics(
    target: np.ndarray, probabilities: np.ndarray, classes: np.ndarray
) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, recall_score

    predicted = classes[np.argmax(probabilities, axis=1)]
    top_k = min(3, classes.size)
    top_indices = np.argpartition(probabilities, -top_k, axis=1)[:, -top_k:]
    top_labels = classes[top_indices]
    return {
        "macro_f1": float(
            f1_score(target, predicted, labels=classes, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(target, predicted, labels=classes, average="macro", zero_division=0)
        ),
        "accuracy": float(accuracy_score(target, predicted)),
        "top3_accuracy": float(np.mean(np.any(top_labels == target[:, None], axis=1))),
    }


def validation_score(task: str, target: np.ndarray, prediction: np.ndarray) -> float:
    if task == "regression":
        return -float(np.sqrt(np.mean(np.square(target - prediction))))
    classes = np.arange(prediction.shape[1], dtype=np.int64)
    return classification_metrics(target, prediction, classes)["macro_f1"]


def model_classes():
    import torch
    from torch import nn

    class ContentMLP(nn.Module):
        def __init__(self, content_features: int, _brain_features: int, outputs: int):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.LayerNorm(content_features),
                nn.Linear(content_features, 128),
                nn.GELU(),
                nn.Dropout(0.30),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Dropout(0.20),
            )
            self.head = nn.Linear(64, outputs)

        def forward(self, content, _brain):
            return self.head(self.encoder(content))

    class GatedMLP(nn.Module):
        def __init__(self, content_features: int, brain_features: int, outputs: int):
            super().__init__()
            self.content_encoder = nn.Sequential(
                nn.LayerNorm(content_features),
                nn.Linear(content_features, 128),
                nn.GELU(),
                nn.Dropout(0.30),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Dropout(0.20),
            )
            self.brain_encoder = nn.Sequential(
                nn.LayerNorm(brain_features),
                nn.Linear(brain_features, 64),
                nn.GELU(),
                nn.Dropout(0.30),
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Dropout(0.20),
            )
            self.content_head = nn.Linear(64, outputs)
            self.correction = nn.Linear(96, outputs)
            self.gate = nn.Linear(64, 1)
            nn.init.zeros_(self.correction.weight)
            nn.init.zeros_(self.correction.bias)
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, -2.0)

        def forward(self, content, brain):
            content_state = self.content_encoder(content)
            brain_state = self.brain_encoder(brain)
            base = self.content_head(content_state)
            correction = self.correction(torch.cat([content_state, brain_state], dim=1))
            return base + torch.sigmoid(self.gate(content_state)) * correction

    class ScalarFeatureTokenizer(nn.Module):
        def __init__(self, features: int, token_width: int):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(features, token_width))
            self.bias = nn.Parameter(torch.empty(features, token_width))
            nn.init.normal_(self.weight, std=0.02)
            nn.init.normal_(self.bias, std=0.02)

        def forward(self, values):
            return values.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)

    class ReducedFTTransformer(nn.Module):
        def __init__(
            self, content_features: int, brain_features: int, outputs: int, combined: bool
        ):
            super().__init__()
            self.combined = combined
            features = content_features + (brain_features if combined else 0)
            width = 32
            self.tokenizer = ScalarFeatureTokenizer(features, width)
            self.cls = nn.Parameter(torch.zeros(1, 1, width))
            layer = nn.TransformerEncoderLayer(
                d_model=width,
                nhead=4,
                dim_feedforward=64,
                dropout=0.20,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=2)
            self.norm = nn.LayerNorm(width)
            self.head = nn.Linear(width, outputs)
            nn.init.normal_(self.cls, std=0.02)

        def forward(self, content, brain):
            values = torch.cat([content, brain], dim=1) if self.combined else content
            tokens = self.tokenizer(values)
            cls = self.cls.expand(values.shape[0], -1, -1)
            encoded = self.transformer(torch.cat([cls, tokens], dim=1))
            return self.head(self.norm(encoded[:, 0]))

    return ContentMLP, GatedMLP, ReducedFTTransformer


def build_factory(
    model_name: str, content_features: int, brain_features: int, outputs: int
) -> Callable[[], Any]:
    ContentMLP, GatedMLP, ReducedFTTransformer = model_classes()
    if model_name == "content_mlp":
        return lambda: ContentMLP(content_features, brain_features, outputs)
    if model_name == "gated_mlp":
        return lambda: GatedMLP(content_features, brain_features, outputs)
    if model_name == "content_ft":
        return lambda: ReducedFTTransformer(
            content_features, brain_features, outputs, combined=False
        )
    if model_name == "combined_ft":
        return lambda: ReducedFTTransformer(
            content_features, brain_features, outputs, combined=True
        )
    raise DeepEvaluationError(f"unknown model: {model_name}")


def batches(size: int, batch_size: int, rng: np.random.Generator):
    order = rng.permutation(size)
    for start in range(0, size, batch_size):
        yield order[start : start + batch_size]


def forward_numpy(model, content, brain, batch_size: int, device):
    import torch

    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, content.shape[0], batch_size):
            end = min(start + batch_size, content.shape[0])
            content_tensor = torch.as_tensor(content[start:end], device=device)
            brain_tensor = torch.as_tensor(brain[start:end], device=device)
            outputs.append(model(content_tensor, brain_tensor).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def target_setup(task: str, target: np.ndarray, device, class_count: int):
    import torch

    if task == "regression":
        center = float(np.mean(target))
        scale = float(np.std(target))
        if scale <= 1e-8:
            raise DeepEvaluationError("training target has zero variance")
        transformed = ((target - center) / scale).astype(np.float32)
        return transformed, torch.nn.SmoothL1Loss(), center, scale
    counts = np.bincount(target, minlength=class_count).astype(np.float64)
    if np.any(counts == 0):
        weights = np.zeros_like(counts)
        present = counts > 0
        weights[present] = np.sqrt(target.size / (present.sum() * counts[present]))
        weights[~present] = 0.0
    else:
        weights = np.sqrt(target.size / (counts.size * counts))
    weights = np.clip(weights, 0.0, 5.0).astype(np.float32)
    loss = torch.nn.CrossEntropyLoss(
        weight=torch.as_tensor(weights, device=device), label_smoothing=0.05
    )
    return target.astype(np.int64), loss, 0.0, 1.0


def train_epochs(
    model,
    content: np.ndarray,
    brain: np.ndarray,
    target: np.ndarray,
    task: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device,
    class_count: int,
    validation: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    patience: int = 0,
) -> tuple[Any, int, float, dict[str, float]]:
    import torch

    model = model.to(device)
    transformed, loss_function, center, scale = target_setup(
        task, target, device, class_count
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    rng = np.random.default_rng(seed)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 1
    best_score = -np.inf
    stale = 0
    last_loss = float("nan")
    for epoch in range(1, epochs + 1):
        model.train()
        cumulative = 0.0
        examples = 0
        for index in batches(content.shape[0], batch_size, rng):
            content_tensor = torch.as_tensor(content[index], device=device)
            brain_tensor = torch.as_tensor(brain[index], device=device)
            if task == "regression":
                target_tensor = torch.as_tensor(transformed[index], device=device).unsqueeze(1)
            else:
                target_tensor = torch.as_tensor(transformed[index], device=device)
            optimizer.zero_grad(set_to_none=True)
            output = model(content_tensor, brain_tensor)
            loss = loss_function(output, target_tensor)
            if not torch.isfinite(loss):
                raise DeepEvaluationError("training produced a non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            cumulative += float(loss.detach().cpu()) * index.size
            examples += index.size
        last_loss = cumulative / examples
        if validation is None:
            continue
        val_content, val_brain, val_target = validation
        raw = forward_numpy(model, val_content, val_brain, batch_size, device)
        if task == "regression":
            prediction = raw[:, 0] * scale + center
        else:
            prediction = torch.softmax(torch.as_tensor(raw), dim=1).numpy()
        score = validation_score(task, val_target, prediction)
        if score > best_score + 1e-7:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch >= 10 and stale >= patience:
            break
    if validation is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, best_score, {
        "last_training_loss": last_loss,
        "target_center": center,
        "target_scale": scale,
    }


def fit_and_predict(
    factory: Callable[[], Any],
    content_train: np.ndarray,
    brain_train: np.ndarray,
    target_train: np.ndarray,
    content_test: np.ndarray,
    brain_test: np.ndarray,
    task: str,
    seed: int,
    args: argparse.Namespace,
    device,
    class_count: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_ids = np.asarray([str(index) for index in range(target_train.size)])
    inner = common.deterministic_group_folds(train_ids, 5, seed + 7000)
    fit_index, validation_index = inner[0]
    set_seed(seed)
    selection_model, best_epoch, best_score, selection_info = train_epochs(
        factory(),
        content_train[fit_index],
        brain_train[fit_index],
        target_train[fit_index],
        task,
        args.max_epochs,
        args.batch_size,
        args.learning_rate,
        args.weight_decay,
        seed,
        device,
        class_count,
        validation=(
            content_train[validation_index],
            brain_train[validation_index],
            target_train[validation_index],
        ),
        patience=args.patience,
    )
    del selection_model
    if str(device).startswith("cuda"):
        import torch

        torch.cuda.empty_cache()
    set_seed(seed)
    final_model, _, _, final_info = train_epochs(
        factory(),
        content_train,
        brain_train,
        target_train,
        task,
        best_epoch,
        args.batch_size,
        args.learning_rate,
        args.weight_decay,
        seed,
        device,
        class_count,
    )
    raw = forward_numpy(final_model, content_test, brain_test, args.batch_size, device)
    if task == "regression":
        prediction = raw[:, 0] * final_info["target_scale"] + final_info["target_center"]
    else:
        import torch

        prediction = torch.softmax(torch.as_tensor(raw), dim=1).numpy()
    if not np.isfinite(prediction).all():
        raise DeepEvaluationError("model produced non-finite test predictions")
    return prediction.astype(np.float64), {
        "seed": seed,
        "best_epoch": int(best_epoch),
        "validation_score": float(best_score),
        "selection": selection_info,
        "refit": final_info,
        "fit_n": int(fit_index.size),
        "validation_n": int(validation_index.size),
    }


def run(args: argparse.Namespace) -> Path:
    if args.fold not in range(5):
        raise DeepEvaluationError("fold must be one of 0,1,2,3,4")
    all_targets = {**REGRESSION_TARGETS, **CLASSIFICATION_TARGETS}
    if args.target_name not in all_targets:
        raise DeepEvaluationError(f"unknown target: {args.target_name}")
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_file = cache_dir / "reduced_features.npz"
    cache_metadata = cache_dir / "metadata.json"
    if not cache_file.is_file() or not cache_metadata.is_file():
        raise DeepEvaluationError(f"cache is incomplete: {cache_dir}")
    with np.load(cache_file, allow_pickle=False) as cache:
        train_indices = cache["train_indices"].astype(np.int64)
        test_indices = cache["test_indices"].astype(np.int64)
        train_ids = cache["train_ids"].astype(str)
        test_ids = cache["test_ids"].astype(str)
        content_train = cache["content_train"].astype(np.float32)
        content_test = cache["content_test"].astype(np.float32)
        brain_train = cache["brain_train"].astype(np.float32)
        brain_test = cache["brain_test"].astype(np.float32)
    ids = np.empty(train_indices.size + test_indices.size, dtype=object)
    ids[train_indices] = train_ids
    ids[test_indices] = test_ids
    target, target_column = load_target(
        Path(args.cohort).expanduser().resolve(), ids.astype(str), args.target_name
    )
    target_train = target[train_indices]
    target_test = target[test_indices]
    task = "classification" if args.target_name in CLASSIFICATION_TARGETS else "regression"
    classes = np.unique(target).astype(np.int64) if task == "classification" else None
    if task == "classification":
        class_to_position = {int(value): index for index, value in enumerate(classes)}
        encoded_train = np.asarray([class_to_position[int(value)] for value in target_train])
        output_count = classes.size
    else:
        encoded_train = target_train.astype(np.float64)
        output_count = 1
    device = resolve_device(args.device)
    if args.smoke_test:
        train_limit = min(args.smoke_train_n, encoded_train.size)
        test_limit = min(args.smoke_test_n, target_test.size)
        content_train = content_train[:train_limit]
        brain_train = brain_train[:train_limit]
        encoded_train = encoded_train[:train_limit]
        target_train = target_train[:train_limit]
        content_test = content_test[:test_limit]
        brain_test = brain_test[:test_limit]
        target_test = target_test[:test_limit]
        test_ids = test_ids[:test_limit]
        args.max_epochs = min(args.max_epochs, 3)
        args.patience = min(args.patience, 2)
        args.seeds = args.seeds[:1]
    print(
        f"DEEP_START target={args.target_name} fold={args.fold} task={task} "
        f"device={device} train={encoded_train.size} test={target_test.size} "
        f"content={content_train.shape[1]} brain={brain_train.shape[1]} "
        f"seeds={args.seeds} smoke={args.smoke_test}",
        flush=True,
    )
    predictions: dict[str, np.ndarray] = {}
    training: dict[str, list[dict[str, Any]]] = {}
    metric_rows: list[dict[str, Any]] = []
    for model_name in MODELS:
        print(f"MODEL_START target={args.target_name} fold={args.fold} model={model_name}", flush=True)
        factory = build_factory(
            model_name, content_train.shape[1], brain_train.shape[1], output_count
        )
        seed_predictions = []
        training[model_name] = []
        for offset, seed_base in enumerate(args.seeds):
            seed = int(seed_base + args.fold * 100 + offset)
            prediction, info = fit_and_predict(
                factory,
                content_train,
                brain_train,
                encoded_train,
                content_test,
                brain_test,
                task,
                seed,
                args,
                device,
                output_count,
            )
            seed_predictions.append(prediction)
            training[model_name].append(info)
            print(
                f"SEED_COMPLETE target={args.target_name} fold={args.fold} "
                f"model={model_name} seed={seed} epoch={info['best_epoch']} "
                f"validation={info['validation_score']:.6f}",
                flush=True,
            )
        averaged = np.mean(np.stack(seed_predictions, axis=0), axis=0)
        predictions[model_name] = averaged
        if task == "regression":
            metrics = common.calculate_metrics(target_test.astype(np.float64), averaged)
        else:
            metrics = classification_metrics(target_test, averaged, classes)
        metric_rows.append({"model": model_name, **metrics})
        primary = metrics["spearman"] if task == "regression" else metrics["macro_f1"]
        print(
            f"MODEL_COMPLETE target={args.target_name} fold={args.fold} "
            f"model={model_name} primary={primary:.6f}",
            flush=True,
        )
    final = Path(args.output_dir).expanduser().resolve()

    def write(staging: Path) -> None:
        if task == "regression":
            common.write_csv(
                staging / "predictions.csv",
                ["target", "fold", "video_id", "observed", *MODELS],
                (
                    {
                        "target": args.target_name,
                        "fold": args.fold,
                        "video_id": str(test_ids[row]),
                        "observed": float(target_test[row]),
                        **{name: float(predictions[name][row]) for name in MODELS},
                    }
                    for row in range(target_test.size)
                ),
            )
        else:
            common.write_csv(
                staging / "predictions.csv",
                [
                    "target", "fold", "video_id", "observed",
                    *[f"{name}__prediction" for name in MODELS],
                ],
                (
                    {
                        "target": args.target_name,
                        "fold": args.fold,
                        "video_id": str(test_ids[row]),
                        "observed": int(target_test[row]),
                        **{
                            f"{name}__prediction": int(classes[np.argmax(predictions[name][row])])
                            for name in MODELS
                        },
                    }
                    for row in range(target_test.size)
                ),
            )
            np.savez_compressed(
                staging / "probabilities.npz",
                video_ids=test_ids,
                observed=target_test.astype(np.int64),
                classes=classes,
                **{name: predictions[name].astype(np.float32) for name in MODELS},
            )
        metric_fields = (
            ["model", *common.METRICS]
            if task == "regression"
            else ["model", "macro_f1", "macro_recall", "accuracy", "top3_accuracy"]
        )
        common.write_csv(staging / "model_metrics.csv", metric_fields, metric_rows)
        common.dump_json(staging / "training_summary.json", training)
        common.dump_json(
            staging / "metadata.json",
            {
                "schema_version": 1,
                "script_version": SCRIPT_VERSION,
                "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "target": args.target_name,
                "target_column": target_column,
                "task": task,
                "fold": args.fold,
                "train_n": int(encoded_train.size),
                "test_n": int(target_test.size),
                "outer_group_overlap": 0,
                "models": list(MODELS),
                "classes": classes.astype(int).tolist() if classes is not None else None,
                "seeds": [int(value) for value in args.seeds],
                "max_epochs": args.max_epochs,
                "patience": args.patience,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "device": str(device),
                "smoke_test": bool(args.smoke_test),
                "cache": {
                    "path": str(cache_dir),
                    "npz_sha256": common.sha256_file(cache_file),
                    "metadata_sha256": common.sha256_file(cache_metadata),
                },
                "cohort": {
                    "path": str(Path(args.cohort).expanduser().resolve()),
                    "sha256": common.sha256_file(Path(args.cohort).expanduser().resolve()),
                },
            },
        )

    atomic_directory(final, write)
    print(f"DEEP_COMPLETE target={args.target_name} fold={args.fold} output={final}", flush=True)
    return final


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--target-name",
        choices=sorted([*REGRESSION_TARGETS, *CLASSIFICATION_TARGETS]),
        required=True,
    )
    result.add_argument("--fold", type=int, required=True)
    result.add_argument("--cohort", required=True)
    result.add_argument("--cache-dir", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    result.add_argument("--seeds", type=int, nargs="+", default=[20260830, 20260930])
    result.add_argument("--max-epochs", type=int, default=100)
    result.add_argument("--patience", type=int, default=12)
    result.add_argument("--batch-size", type=int, default=128)
    result.add_argument("--learning-rate", type=float, default=3e-4)
    result.add_argument("--weight-decay", type=float, default=1e-3)
    result.add_argument("--smoke-test", action="store_true")
    result.add_argument("--smoke-train-n", type=int, default=256)
    result.add_argument("--smoke-test-n", type=int, default=64)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    output = run(args)
    print(f"OUTPUT {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
