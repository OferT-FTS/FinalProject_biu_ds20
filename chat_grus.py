Nice — below is a **single, runnable end-to-end pipeline** that:

* Loads transaction CSVs (expects `card_id`, `transaction_time`, `is_fraud` plus feature columns),
* Splits data **by card** (no leakage) into train/val/test,
* Fits a `ColumnTransformer` (scales numeric, one-hot encodes categoricals) **on training data only** and persists it,
* Builds fixed-length sliding sequences per card (drop histories shorter than `seq_len`),
* Trains a **CuDNN-optimized GRU** stacked model with **focal loss + class weights**,
* Evaluates (PR AUC + classification report),
* Saves model + transformers, and prints model summary (so you can inspect params).

I kept the code modular and documented. Copy it into a file (e.g. `fraud_pipeline.py`) and run.

---

## Requirements

```bash
pip install tensorflow pandas numpy scikit-learn pydantic joblib
```

(If you have GPU and compatible TF, GRU will use CuDNN automatically. If not, TF falls back to CPU GRU.)

---

## File: `fraud_pipeline.py`

```python
"""
fraud_pipeline.py
End-to-end stacked GRU fraud detection pipeline with preprocessing, training and evaluation.
"""

import os
import time
import joblib
import logging
from typing import List, Tuple, Any, Dict

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, precision_recall_curve, auc
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

# ----------------------------
# Config (Pydantic)
# ----------------------------
class AppConfig(BaseModel):
    data_path: Path = Field(default_factory=lambda: Path("transactions.csv"))
    project_root: Path = Field(default_factory=Path.cwd)

    # columns - update to match your CSV
    id_col: str = "card_id"
    time_col: str = "transaction_time"
    label_col: str = "is_fraud"

    # feature columns: list numeric and categorical; adjust to your dataset
    numeric_cols: List[str] = Field(default_factory=lambda: ["amount", "balance"])
    categorical_cols: List[str] = Field(default_factory=lambda: ["merchant_id", "device_type"])

    # sequence/model params
    seq_len: int = 10
    n_features: int = None  # set automatically after transformer fit
    batch_size: int = 64
    epochs: int = 10
    seed: int = 42

    # output
    model_dir: Path = Field(default_factory=lambda: Path("models"))
    transformer_path: Path = Field(default_factory=lambda: Path("models/transformer.joblib"))
    model_name: str = "stacked_gru_focal"

    # training split
    train_frac: float = 0.7
    val_frac: float = 0.15
    test_frac: float = 0.15

    class Config:
        frozen = True
        extra = "forbid"


# ----------------------------
# BaseComponent with timer and logger
# ----------------------------
class BaseComponent:
    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = logging.getLogger("fraud_pipeline")
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            self.logger.addHandler(ch)
            self.logger.setLevel(logging.INFO)

    def timer(self, label: str = "Task"):
        return self._Timer(self.logger, label)

    class _Timer:
        def __init__(self, logger, label):
            self.logger = logger
            self.label = label

        def __enter__(self):
            self.start = time.time()
            self.logger.info(f"[{self.label}] started")

        def __exit__(self, exc_type, exc_val, exc_tb):
            elapsed = time.time() - self.start
            self.logger.info(f"[{self.label}] completed in {elapsed:.2f}s")


# ----------------------------
# Preprocessing utilities
# ----------------------------
def load_data(path: Path, time_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # ensure time column parsed
    if not np.issubdtype(df[time_col].dtype, np.datetime64):
        df[time_col] = pd.to_datetime(df[time_col])
    return df


def split_by_group(df: pd.DataFrame, id_col: str, train_frac: float, val_frac: float, seed: int) -> Tuple[pd.Index, pd.Index, pd.Index]:
    """
    Split unique ids into train/val/test sets using GroupShuffleSplit.
    Returns three Index objects of ids.
    """
    unique_ids = df[id_col].unique()
    gss = GroupShuffleSplit(n_splits=1, train_size=train_frac, random_state=seed)
    train_idx, temp_idx = next(gss.split(unique_ids, groups=unique_ids))
    train_ids = unique_ids[train_idx]
    temp_ids = unique_ids[temp_idx]

    # split temp into val/test equally according to val_frac relative to total
    relative_val = val_frac / (1 - train_frac)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=relative_val, random_state=seed)
    val_idx, test_idx = next(gss2.split(temp_ids, groups=temp_ids))
    val_ids = temp_ids[val_idx]
    test_ids = temp_ids[test_idx]

    return pd.Index(train_ids), pd.Index(val_ids), pd.Index(test_ids)


def fit_transformer(train_df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """
    Fit a ColumnTransformer on training rows and return it.
    Numeric -> StandardScaler, Categorical -> OneHotEncoder(handle_unknown='ignore', sparse=False)
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)

    transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ],
        remainder="drop",  # drop any other columns
        sparse_threshold=0
    )
    transformer.fit(train_df[numeric_cols + categorical_cols])
    return transformer


def transform_rows(transformer: ColumnTransformer, df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> np.ndarray:
    cols = numeric_cols + categorical_cols
    X = transformer.transform(df[cols])
    return np.asarray(X, dtype=np.float32)


def build_sequences_from_transformed(df: pd.DataFrame, id_col: str, time_col: str, features_array: np.ndarray, label_col: str, seq_len: int
                                    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sliding-window sequences from transformed features_array (shape [n_rows, n_features_transformed]).
    - df must be sorted by id_col and time_col and align with features_array rows (i.e. same index order).
    Returns X (N, seq_len, n_features) and y (N,)
    """
    assert len(df) == features_array.shape[0]
    X_list, y_list = [], []
    grouped = df.groupby(id_col, sort=False)
    row_offset = 0  # to index into features_array
    for _, g in grouped:
        n = len(g)
        if n <= seq_len:
            row_offset += n
            continue
        # for each transaction that has seq_len historic rows before it
        for end_idx in range(seq_len, n):
            start_idx = end_idx - seq_len
            # rows belong to global features_array, from row_offset + start_idx to row_offset + end_idx - 1
            s = row_offset + start_idx
            e = row_offset + end_idx
            seq = features_array[s:e]  # shape (seq_len, n_features)
            X_list.append(seq)
            y_list.append(g.iloc[end_idx][label_col])
        row_offset += n
    if len(X_list) == 0:
        return np.empty((0, seq_len, features_array.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int32)
    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y


# ----------------------------
# Model (GRU + focal loss)
# ----------------------------
def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        # stable binary crossentropy
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        loss_val = alpha * tf.pow((1 - p_t), gamma) * bce
        return tf.reduce_mean(loss_val)
    return loss


def build_stacked_gru(seq_len: int, n_features: int) -> tf.keras.Model:
    inp = layers.Input(shape=(seq_len, n_features), name="seq_input")
    x = layers.GRU(128, return_sequences=True)(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.GRU(64, return_sequences=True)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.GRU(32, return_sequences=True)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.GRU(16, return_sequences=False)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(8, activation="relu")(x)
    x = layers.Dense(4, activation="relu")(x)
    x = layers.Dense(2, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=inp, outputs=out, name="stacked_gru_focal")
    model.compile(optimizer=optimizers.Adam(), loss=focal_loss(alpha=0.25, gamma=2.0),
                  metrics=[tf.keras.metrics.AUC(name="auc"),
                           tf.keras.metrics.Precision(name="precision"),
                           tf.keras.metrics.Recall(name="recall")])
    return model


# ----------------------------
# Trainer class that uses BaseComponent
# ----------------------------
class FraudPipeline(BaseComponent):
    def __init__(self, config: AppConfig):
        super().__init__(config)
        self.config = config
        os.makedirs(self.config.model_dir, exist_ok=True)

    def prepare(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Splits data by card_id, fits transformer on train rows, transforms all rows, builds sequences for each split.
        Returns dict with X_train, y_train, X_val, y_val, X_test, y_test and fitted transformer.
        """
        cfg = self.config
        self.logger.info("Sorting data by id and time")
        df_sorted = df.sort_values([cfg.id_col, cfg.time_col]).reset_index(drop=True)

        # split ids
        self.logger.info("Splitting card ids into train/val/test (grouped split)")
        train_ids, val_ids, test_ids = split_by_group(df_sorted, cfg.id_col, cfg.train_frac, cfg.val_frac, cfg.seed)
        self.logger.info(f"Counts: train_ids={len(train_ids)}, val_ids={len(val_ids)}, test_ids={len(test_ids)}")

        # select rows for fitting transformer (train rows only)
        train_rows_mask = df_sorted[cfg.id_col].isin(train_ids)
        train_rows = df_sorted[train_rows_mask]

        # fit transformer
        self.logger.info("Fitting ColumnTransformer on train rows")
        transformer = fit_transformer(train_rows, cfg.numeric_cols, cfg.categorical_cols)

        # transform all rows (after fitting on train only)
        self.logger.info("Transforming entire dataset using fitted transformer")
        features_all = transform_rows(transformer, df_sorted, cfg.numeric_cols, cfg.categorical_cols)
        cfg_dict = cfg.dict()
        # set n_features in config-like dict for downstream use
        n_features_transformed = features_all.shape[1]
        self.logger.info(f"Transformed feature dimension: {n_features_transformed}")

        # build sequences per split
        self.logger.info("Building sequences for train split")
        train_mask = df_sorted[cfg.id_col].isin(train_ids)
        X_train, y_train = build_sequences_from_transformed(df_sorted[train_mask], cfg.id_col, cfg.time_col,
                                                            features_all[train_mask.values], cfg.label_col, cfg.seq_len)

        self.logger.info("Building sequences for val split")
        val_mask = df_sorted[cfg.id_col].isin(val_ids)
        X_val, y_val = build_sequences_from_transformed(df_sorted[val_mask], cfg.id_col, cfg.time_col,
                                                        features_all[val_mask.values], cfg.label_col, cfg.seq_len)

        self.logger.info("Building sequences for test split")
        test_mask = df_sorted[cfg.id_col].isin(test_ids)
        X_test, y_test = build_sequences_from_transformed(df_sorted[test_mask], cfg.id_col, cfg.time_col,
                                                          features_all[test_mask.values], cfg.label_col, cfg.seq_len)

        self.logger.info(f"Sequence shapes: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")

        result = {
            "transformer": transformer,
            "n_features_transformed": n_features_transformed,
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test
        }
        return result

    def compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        if len(y) == 0:
            return {}
        classes = np.unique(y)
        cw = class_weight.compute_class_weight('balanced', classes=classes, y=y)
        return {int(c): float(w) for c, w in zip(classes, cw)}

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, n_features: int):
        cfg = self.config
        model = build_stacked_gru(cfg.seq_len, n_features)
        model.summary(print_fn=self.logger.info)

        cw = self.compute_class_weights(y_train)
        self.logger.info(f"Class weights used: {cw}")

        model_path = os.path.join(cfg.model_dir, f"{cfg.model_name}.h5")
        cb = [
            callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=5, restore_best_weights=True),
            callbacks.ModelCheckpoint(model_path, monitor="val_auc", mode="max", save_best_only=True),
            callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=2)
        ]
        with self.timer("Model Training"):
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val) if len(X_val) > 0 else None,
                epochs=cfg.epochs,
                batch_size=cfg.batch_size,
                class_weight=cw if cw else None,
                callbacks=cb,
                verbose=2
            )
        self.logger.info(f"Model saved to {model_path}")
        return model, history.history

    def evaluate(self, model, X_test: np.ndarray, y_test: np.ndarray):
        self.logger.info("Evaluating on test set")
        if len(X_test) == 0:
            self.logger.warning("No test sequences available")
            return {}
        probs = model.predict(X_test, batch_size=self.config.batch_size).ravel()
        preds = (probs >= 0.5).astype(int)
        self.logger.info("Classification report:\n" + classification_report(y_test, preds, digits=4))
        precision, recall, _ = precision_recall_curve(y_test, probs)
        pr_auc = auc(recall, precision)
        self.logger.info(f"PR AUC: {pr_auc:.4f}")
        return {"probs": probs, "pr_auc": pr_auc}


# ----------------------------
# Main execution
# ----------------------------
def main():
    cfg = AppConfig()
    pipeline = FraudPipeline(cfg)

    pipeline.logger.info(f"Loading data from {cfg.data_path}")
    df = load_data(cfg.data_path, cfg.time_col)

    with pipeline.timer("Preprocessing & sequence building"):
        prep = pipeline.prepare(df)

    # persist transformer
    pipeline.logger.info(f"Saving transformer to {cfg.transformer_path}")
    joblib.dump(prep["transformer"], cfg.transformer_path)

    n_features = prep["n_features_transformed"]
    X_train, y_train = prep["X_train"], prep["y_train"]
    X_val, y_val = prep["X_val"], prep["y_val"]
    X_test, y_test = prep["X_test"], prep["y_test"]

    # quick sanity checks
    pipeline.logger.info(f"Train sequences: {X_train.shape}, positives: {y_train.sum()}")
    pipeline.logger.info(f"Val sequences: {X_val.shape}, positives: {y_val.sum() if len(y_val)>0 else 'n/a'}")
    pipeline.logger.info(f"Test sequences: {X_test.shape}, positives: {y_test.sum() if len(y_test)>0 else 'n/a'}")

    model, history = pipeline.train(X_train, y_train, X_val, y_val, n_features)
    eval_results = pipeline.evaluate(model, X_test, y_test)

    pipeline.logger.info("Pipeline finished.")

if __name__ == "__main__":
    main()
```

---

## How to use

1. Prepare your CSV `transactions.csv` (or update `data_path` in `AppConfig`) with these columns:

   * `card_id` — unique identifier for card/customer,
   * `transaction_time` — ISO datetime (will be parsed),
   * `is_fraud` — 0/1 label,
   * numeric feature columns e.g. `amount`, `balance`,
   * categorical columns e.g. `merchant_id`, `device_type`.
     Edit `numeric_cols` and `categorical_cols` in the `AppConfig` section if your column names differ.

2. Run:

```bash
python fraud_pipeline.py
```

3. Outputs:

   * Saved model: `models/stacked_gru_focal.h5`
   * Saved transformer: `models/transformer.joblib`
   * Console logs include model summary, parameter counts, class weights, PR AUC, and classification report.

---

## Notes, trade-offs & extensions

* **Feature engineering**: current pipeline uses raw numeric + one-hot categorical. For better results:

  * Add engineered features (normalized amount, time-of-day, transaction velocity).
  * Replace OneHotEncoder with learned embeddings (requires mapping categories to integer IDs and an embedding layer).
* **Sequence length & padding**: this pipeline drops groups shorter than `seq_len`. If you want to include shorter histories, implement padding + `Masking` layer.
* **Leakage avoidance**: we split by `card_id` (grouped split) to avoid leakage; this is important.
* **Imbalance**: we use both class weights and focal loss. You can also experiment with oversampling sequences for fraud or using advanced loss/metrics.
* **Scaling & persistence**: `ColumnTransformer` is fit on training rows only and saved with `joblib` for production inference.
* **Production inference**: to score a new transaction, retrieve last `seq_len` transactions for the card, transform using `transformer.transform`, shape to `(1, seq_len, n_features)`, and call `model.predict`.
* **GPU acceleration**: if TensorFlow detects GPU and compatible cuDNN, GRU will use cuDNN kernels (very fast). If not, it runs CPUGRU.

---

If you want, I can:

* adapt this to use padding + masking so you don't drop short histories,
* replace `OneHotEncoder` with categorical `OrdinalEncoder + Embedding` and modify the model to include embedding layers for large-cardinality columns,
* add more advanced monitoring (memory/CPU logging during training),
* convert to a reusable package with CLI args or a config YAML, or
* provide a notebook example integrating a real (anonymized) sample dataset.

Which of these should I do next?







Short answer: **SMOTE is *not* recommended for your sequence-based GRU fraud pipeline**, and in most cases it will *hurt* performance rather than help.

Below is the clear explanation why, plus what you *should* do instead.

---

# ✅ Why SMOTE is *not useful* for the GRU/sequence fraud pipeline

Your model does **not** train on independent rows.
It trains on **sequences of transactions**:

```
Seq 1: [x1, x2, x3, ..., x10] → label
Seq 2: [x2, x3, x4, ..., x11] → label
...
```

SMOTE works by:

* interpolating between **feature vectors** in feature space,
* creating **synthetic standalone samples**.

### 🔴 Issue 1 — SMOTE destroys temporal structure

SMOTE mixes two transactions into a new "synthetic" one:

```
x_synthetic = x_i + α (x_j – x_i)
```

But your model expects **coherent temporal sequences**, not standalone interpolated points.
Synthetic samples created by SMOTE would:

* break transaction order,
* produce unrealistic time series transitions,
* distort patterns in spending velocity, amount spikes, etc.

This leads to **nonsense sequences** → degraded model performance.

### 🔴 Issue 2 — SMOTE assumes IID (independent rows)

Sequence models assume *ordered* data within each card.
SMOTE assumes *unordered* IID data.
They are mathematically incompatible.

### 🔴 Issue 3 — Fraud data is *not interpolatable*

Fraudulent transactions have *intentional anomalous patterns*.
Interpolating:

```
fraud + nonfraud → synthetic sample
```

produces *unrealistic non-fraud-like points* which weaken the signal.

---

# 🚫 So for your GRU fraud pipeline:

❌ Do **NOT** apply SMOTE before sequence generation
❌ Do **NOT** apply SMOTE after sequence generation (makes sequences inconsistent)

---

# ✅ What *does* work for heavy fraud imbalance (and is already included)

### 1. **Class weights** ✔️ (included)

This scales the loss but does not change the data.

### 2. **Focal loss** ✔️ (included)

This helps the model focus on rare fraud cases.

### 3. **Temporal oversampling of *whole sequences*** (safe alternative)

Instead of interpolating rows (SMOTE), you can **duplicate minority sequences**:

* If fraudulent sequences are < 1%, oversample them by 2–10×
* Always oversample **entire sequences**, not individual rows

This keeps temporal structure intact.

I can add this to the pipeline if you want.

---

# ⭐ Best practices for sequence models in fraud detection

### Recommended (safe)

| Method                      | Works? | Notes                     |
| --------------------------- | ------ | ------------------------- |
| Class weights               | ✔️     | Already used              |
| Focal loss                  | ✔️     | Already used              |
| Sequence-level oversampling | ✔️     | Duplicate whole sequences |
| Cost-sensitive thresholds   | ✔️     | Tune decision threshold   |
| Anomaly-focused features    | ✔️     | Rolling z-score, velocity |

### Not recommended (unsafe)

| Method                       | Why it breaks                             |
| ---------------------------- | ----------------------------------------- |
| **SMOTE**                    | Breaks temporal order, unrealistic points |
| ADASYN                       | Same issue as SMOTE                       |
| Random oversampling per row  | Breaks sequence integrity                 |
| Undersampling majority class | Removes valuable normal patterns          |

---

# 🧪 If you still want balancing:

I can show you how to add **sequence-level oversampling**:

```python
def oversample_minority_sequences(X, y, factor=5):
    mask = (y == 1)
    X_min = X[mask]
    y_min = y[mask]
    X_aug = np.repeat(X_min, factor, axis=0)
    y_aug = np.repeat(y_min, factor, axis=0)
    X_new = np.concatenate([X, X_aug])
    y_new = np.concatenate([y, y_aug])
    return X_new, y_new
```

This preserves the time-series structure.

---

# If you want, I can:

✅ Add sequence oversampling directly into the pipeline
or
✅ Modify the pipeline to support embedding + tabular features
or
✅ Show you the correct way to tune the classification threshold for fraud detection (very important!)

Which option would you like?
