"""PyTorch training loop shared by 02_train_arr_model and 03_train_afib_model.

Trains, predicts, and describes a model's layers (``architecture_summary``, logged to MLflow by
the training notebooks). The model classes and their input reshape stay inline in the
notebooks, because MLflow pickles them by value and the serving endpoints do not have
``common``; nothing from this module ends up inside a logged model. torch is imported inside
the functions, so importing this module needs only numpy.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np


class EpochMetrics(NamedTuple):
    """One epoch of training, as logged to MLflow (step = ``epoch``)."""
    epoch: int
    train_loss: float       # mean training loss over the epoch (with label smoothing, if any)
    val_loss: float         # plain cross-entropy on the validation split
    val_accuracy: float     # argmax accuracy on the validation split


def fit(
    build_model: Callable[[], Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    seed: int,
    device: str = "cpu",
    label_smoothing: float = 0.0,
    on_epoch: Callable[[EpochMetrics], None] | None = None,
) -> tuple[Any, list[EpochMetrics]]:
    """Build a model (after seeding torch, so weight init is reproducible) and train it with Adam
    and cross-entropy on (x_train, y_train). After every epoch, the validation loss and accuracy
    are computed, passed to ``on_epoch`` (e.g. an MLflow logger) and returned in the history.

    ``x_*``: float32 (n, n_leads, n_samples); ``y_*``: int64 class index (n,)."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    val_loss_fn = nn.CrossEntropyLoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed),
    )
    y_val_t = torch.from_numpy(y_val)
    history = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = train_loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        val_logits = torch.from_numpy(predict_logits(model, x_val, device=device))
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=total / len(x_train),
            val_loss=float(val_loss_fn(val_logits, y_val_t)),
            val_accuracy=float((val_logits.argmax(1) == y_val_t).float().mean()),
        )
        history.append(metrics)
        if on_epoch is not None:
            on_epoch(metrics)
    return model, history


def predict_logits(model: Any, x: np.ndarray, device: str = "cpu", batch_size: int = 256) -> np.ndarray:
    """(n, n_classes) logits for float32 ``x``, in eval mode, in batches."""
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            out.append(model(torch.from_numpy(x[start:start + batch_size]).to(device)).cpu().numpy())
    return np.concatenate(out)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Row-wise softmax of ``logits / temperature``."""
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = np.exp(z - z.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


def architecture_summary(model: Any, input_shape: tuple[int, ...]) -> tuple[str, int]:
    """Markdown table of ``model``'s layers in the order they run, each with its output shape and
    number of learned parameters, from one forward pass of zeros shaped ``input_shape`` (e.g.
    ``(1, 12, 2500)``), plus the model's total number of trainable parameters. A dependency-free
    stand-in for torchinfo; the model's train/eval mode is restored afterwards."""
    import torch

    def shape(dims: Any) -> str:
        return "(" + ", ".join(str(d) for d in dims) + ")"

    layers: list[tuple[str, Any, tuple[int, ...]]] = []
    hooks = [module.register_forward_hook(
                 lambda mod, _inp, out, name=name: layers.append((name, mod, tuple(out.shape))))
             for name, module in model.named_modules() if not list(module.children())]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(torch.zeros(input_shape, dtype=torch.float32))
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lines = [f"**{type(model).__name__}**: {total:,} trainable parameters; input {shape(input_shape)} "
             "= (batch, leads, samples).", "",
             "| # | layer | module | output shape | parameters |", "|---|---|---|---|---|",
             f"| 0 | input | | {shape(input_shape)} | 0 |"]
    for i, (name, module, out_shape) in enumerate(layers, start=1):
        n_params = sum(p.numel() for p in module.parameters(recurse=False))
        lines.append(f"| {i} | `{name}` | `{module!r}` | {shape(out_shape)} | {n_params:,} |")
    return "\n".join(lines) + "\n", total
