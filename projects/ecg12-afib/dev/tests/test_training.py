"""Tests for the training-side shared code: common/training.py (torch: training loop, architecture
summary), common/training_data.py and common/ecg_plots.py. torch- and Spark-dependent tests skip
without them."""
import numpy as np
import pytest

from common import config, ecg_plots, training, training_data
from common.ecg_signal_gen import AFIB, DERIVATION, DERIVED_LEADS, LABEL_NAMES, LEADS_12, NSR, OTHER, describe_lead

SMALL = {"train": (12, 1), "val": (6, 2), "holdout": (6, 3)}


def _toy_data(n, seed):
    """Two classes a tiny CNN separates in a few epochs: class 1 has 4x the amplitude."""
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 2
    x = (rng.normal(size=(n, 12, 2500)) * (1 + 3 * y)[:, None, None]).astype(np.float32)
    return x, y.astype(np.int64)


def _toy_model():
    import torch.nn as nn
    return nn.Sequential(nn.Conv1d(12, 4, 7, stride=4), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(4, 2))


def test_fit_logs_every_epoch_and_is_reproducible():
    pytest.importorskip("torch")
    x_tr, y_tr = _toy_data(32, 0)
    x_va, y_va = _toy_data(16, 1)
    seen = []
    kwargs = dict(learning_rate=1e-2, epochs=4, batch_size=8, seed=3)
    model, history = training.fit(_toy_model, x_tr, y_tr, x_va, y_va, on_epoch=seen.append, **kwargs)
    assert [m.epoch for m in history] == [0, 1, 2, 3] and seen == history
    assert history[-1].train_loss < history[0].train_loss
    assert all(0.0 <= m.val_accuracy <= 1.0 and m.val_loss > 0 for m in history)
    logits = training.predict_logits(model, x_va, batch_size=5)                  # batches of 5, 5, 5, 1
    assert logits.shape == (16, 2)
    again, history_again = training.fit(_toy_model, x_tr, y_tr, x_va, y_va, **kwargs)
    assert [m.train_loss for m in history_again] == [m.train_loss for m in history]   # same seed, same run
    np.testing.assert_allclose(training.predict_logits(again, x_va), logits, rtol=1e-5)


def test_architecture_summary_lists_every_layer_with_shapes_and_parameters():
    pytest.importorskip("torch")
    model = _toy_model()
    model.train()
    table, total = training.architecture_summary(model, (1, 12, 2500))
    assert total == sum(p.numel() for p in model.parameters()) == 12 * 4 * 7 + 4 + 4 * 2 + 2
    assert model.training                                                         # mode restored
    rows = [line for line in table.splitlines() if line.startswith("| ") and not line.startswith("| #")]
    assert len(rows) == 1 + 5                                                     # input + 5 leaf layers
    assert rows[0] == "| 0 | input | | (1, 12, 2500) | 0 |"
    assert rows[1] == "| 1 | `0` | `Conv1d(12, 4, kernel_size=(7,), stride=(4,))` | (1, 4, 624) | 340 |"
    assert rows[-1].endswith("| (1, 2) | 10 |") and f"{total:,} trainable parameters" in table


def test_softmax_and_temperature_keep_the_decision():
    logits = np.array([[2.0, -1.0], [0.1, 0.3], [-4.0, 4.0]])
    p, soft = training.softmax(logits), training.softmax(logits, temperature=2.0)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    np.testing.assert_allclose(soft.sum(axis=1), 1.0)
    assert (p.argmax(1) == soft.argmax(1)).all() and soft[2, 1] < p[2, 1]          # softer, same call


def test_generated_splits_rows_and_schema():
    batches = training_data.generate_splits(SMALL, frac_afib=1 / 3, frac_other=1 / 3, noise_level=1.0, n_leads=12)
    assert list(batches) == list(SMALL)
    for split, batch in batches.items():
        assert batch.ecg.shape == (SMALL[split][0], 12, 2500)
        assert [int((batch.labels == c).sum()) for c in (NSR, AFIB, OTHER)] == [SMALL[split][0] // 3] * 3
    again = training_data.generate_splits(SMALL, frac_afib=1 / 3, frac_other=1 / 3, noise_level=1.0, n_leads=12)
    np.testing.assert_array_equal(again["holdout"].ecg, batches["holdout"].ecg)    # seeded per split
    assert not np.array_equal(batches["val"].ecg[0], batches["holdout"].ecg[0])     # independent draws
    rows = training_data.to_pandas(batches)
    assert len(rows) == 24 and rows[config.SAMPLE_ID_COL].is_unique
    assert rows[config.SAMPLE_ID_COL].iloc[0] == "train-00000" and set(rows[config.SPLIT_COL]) == set(SMALL)
    assert set(rows["rhythm_name"]) == set(LABEL_NAMES.values())
    pytest.importorskip("pyspark.sql.types")
    schema = training_data.spark_schema(LEADS_12)
    assert [f.name for f in schema.fields] == list(rows.columns)
    comments = {f.name: f.metadata["comment"] for f in schema.fields}
    assert comments["ecg3"] == describe_lead("III") == "Lead III (derived: III = II - I), mV, 2500 samples at 250 Hz"
    assert comments["ecg7"].startswith("Lead V1 (recorded)")


def test_derivation_text_covers_exactly_the_derived_leads():
    assert set(DERIVATION) == set(DERIVED_LEADS)


def test_load_split_reads_one_split_at_a_version(spark):
    from types import SimpleNamespace
    batches = training_data.generate_splits(SMALL, frac_afib=1 / 3, frac_other=1 / 3, noise_level=1.0, n_leads=12)
    sdf = spark.createDataFrame(training_data.to_pandas(batches), schema=training_data.spark_schema(LEADS_12))
    queries = []
    fake = SimpleNamespace(sql=lambda q: queries.append(q) or sdf)
    holdout = training_data.load_split(fake, "c.s.training_data_12l", 3, "holdout")
    assert queries == ["SELECT * FROM c.s.training_data_12l VERSION AS OF 3"]
    assert holdout.x.shape == (6, 12, 2500) and holdout.x.dtype == np.float32
    assert holdout.sample_ids == [f"holdout-{i:05d}" for i in range(6)]
    np.testing.assert_array_equal(holdout.labels, batches["holdout"].labels)
    np.testing.assert_allclose(holdout.x, batches["holdout"].ecg)
    with pytest.raises(ValueError, match="no rows"):
        training_data.load_split(fake, "c.s.training_data_12l", 3, "missing")


def test_showcase_plots():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    batch = training_data.generate_splits({"x": (6, 5)}, frac_afib=1 / 3, frac_other=1 / 3, noise_level=1.0,
                                          n_leads=12)["x"]
    strips = ecg_plots.rhythm_strips(batch.ecg, batch.labels, batch.lead_names, batch.heart_rate_bpm)
    titles = [ax.get_title(loc="left") for ax in strips.axes]
    assert [t.split(",")[0] for t in titles] == list(LABEL_NAMES.values()) and all("bpm" in t for t in titles)
    twelve = ecg_plots.twelve_lead(batch.ecg[0], batch.lead_names, int(batch.labels[0]))
    labels = [t.get_text() for ax in twelve.axes if ax.get_visible() for t in ax.texts]
    assert len(labels) == 12 and sum(lab.endswith("(derived)") for lab in labels) == 4
    eight = training_data.generate_splits({"x": (3, 5)}, frac_afib=1 / 3, frac_other=1 / 3, noise_level=0.0,
                                          n_leads=8)["x"]
    assert sum(ax.get_visible() for ax in ecg_plots.twelve_lead(eight.ecg[0], eight.lead_names, NSR).axes) == 8
    with pytest.raises(ValueError):
        ecg_plots.rhythm_strips(batch.ecg[:0], batch.labels[:0], batch.lead_names)
    plt.close("all")
