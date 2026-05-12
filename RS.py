from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from MLP import get_mlp_param, create_mlp_model
from LSTM import get_lstm_param, create_lstm_model
from DNN import get_dnn_param, create_dnn_model
from CNN import get_cnn_param, create_cnn_model
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping
import gc
import itertools
import random
import os
import numpy as np
import tensorflow as tf
from eval import _build_automl_model, _get_training_and_validation_split


MODEL_TYPE_MAP = {
    0: "MLP",
    1: "CNN",
    2: "LSTM",
    3: "RNN",
    4: "DNN",
}

_AUTO_BATCH_SIZES = [16, 32, 64, 128]
_AUTO_KERNEL_SIZES = [3, 5, 7]
_AUTO_BOUNDS = {
    "learning_rate": (1e-5, 1e-2),
    "neurons": (16, 512),
    "filters": (16, 256),
    "units": (16, 512),
}


def _set_global_seed(seed=None):
    """Set Python/NumPy/TensorFlow seeds for reproducible optimization runs."""
    if seed is None:
        return

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        try:
            tf.random.set_seed(seed)
        except Exception:
            pass

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def _sample_automl_params(rng):
    return {
        "model_type": rng.choice([0, 1, 2, 3, 4]),
        "learning_rate": rng.uniform(_AUTO_BOUNDS["learning_rate"][0], _AUTO_BOUNDS["learning_rate"][1]),
        "batch_size": rng.choice(_AUTO_BATCH_SIZES),
        "epochs": 100,
        "neurons": rng.randint(_AUTO_BOUNDS["neurons"][0], _AUTO_BOUNDS["neurons"][1] + 1),
        "filters": rng.randint(_AUTO_BOUNDS["filters"][0], _AUTO_BOUNDS["filters"][1] + 1),
        "kernel_size": rng.choice(_AUTO_KERNEL_SIZES),
        "units": rng.randint(_AUTO_BOUNDS["units"][0], _AUTO_BOUNDS["units"][1] + 1),
        "n_conv_layers": rng.choice([1, 2, 3]),
        "pool_size": rng.choice([2, 4]),
        "n_dense_layers": rng.choice([1, 2, 3]),
        "dense_units": rng.choice([32, 64, 128, 256]),
        "dropout_rate": rng.choice([0.0, 0.1, 0.2, 0.3, 0.5]),
        "optimizer": rng.randint(0, 2),
        "activation": rng.choice(['relu', 'tanh', 'sigmoid', 'leaky_relu']),
    }


def _iter_cv_splits(obj, n_folds=3):
    if getattr(obj, "use_cv", False) and getattr(obj, "cv_indices", None):
        X_pool = np.asarray(obj.X_train_val)
        y_pool = np.asarray(obj.y_train_val).astype(int)
        for train_idx, val_idx in obj.cv_indices[:n_folds]:
            yield X_pool[train_idx], X_pool[val_idx], y_pool[train_idx], y_pool[val_idx]
    else:
        X_train, X_val, y_train, y_val = _get_training_and_validation_split(obj)
        yield X_train, X_val, y_train, y_val


def _score_automl_params(obj, cfg, n_folds=3, epochs=50):
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=0
    )
    scores = []
    for X_train, X_val, y_train, y_val in _iter_cv_splits(obj, n_folds=n_folds):
        build_cfg = dict(cfg)
        if "optimizer" in build_cfg and "optimizer_idx" not in build_cfg:
            build_cfg["optimizer_idx"] = build_cfg["optimizer"]
        model, x_train, x_val = _build_automl_model(obj, build_cfg, X_train=X_train, X_val=X_val)
        model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=cfg["batch_size"],
            callbacks=[early_stopping],
            verbose=0
        )
        y_pred = model.predict(x_val, verbose=0, batch_size=1024)
        if obj.n_classes == 2:
            y_pred_classes = (y_pred > 0.5).astype(int).flatten()
        else:
            y_pred_classes = np.argmax(y_pred, axis=1)
        scores.append(f1_score(y_val, y_pred_classes, average='weighted'))
        tf.keras.backend.clear_session()
        gc.collect()
    return float(np.mean(scores))


def randomized_search_optimization(obj, testing_model='AUTOML', seed=None):
    _set_global_seed(seed)

    rng = random.Random(seed)
    n_iter = int(getattr(obj, "rs_n_iter", 10))
    n_folds = int(getattr(obj, "rs_n_folds", getattr(obj, "cv_folds", 3)))
    epochs = int(getattr(obj, "fixed_epochs", 50))
    batch_size = int(getattr(obj, "rs_batch_size", 5))

    print("=" * 50)
    print("        RANDOMIZED SEARCH CONFIGURATION")
    print("=" * 50)
    print(f"  Iterations (random samples): {n_iter}")
    print(f"  Cross-validation folds     : {n_folds}")
    print(f"  Scoring metric             : f1_weighted")
    print(f"  Batch size (n per gen)      : {batch_size}")
    print("=" * 50)

    results = []
    best_by_model = {}
    best_score = -1.0
    best_params = None

    for i in range(n_iter):
        cfg = _sample_automl_params(rng)
        score = _score_automl_params(obj, cfg, n_folds=n_folds, epochs=epochs)
        results.append({"params": cfg, "mean_test_score": score})

        mtype = int(cfg["model_type"])
        prev = best_by_model.get(mtype)
        if prev is None or score > prev["score"]:
            best_by_model[mtype] = {"params": cfg, "score": score}

        if score > best_score:
            best_score = score
            best_params = cfg

        print(f"Sample {i + 1}/{n_iter} | model={MODEL_TYPE_MAP.get(mtype)} | mean_f1={score:.4f}")
        if batch_size > 0 and (i + 1) % batch_size == 0:
            gen_idx = (i + 1) // batch_size
            start = i + 1 - batch_size
            end = i + 1
            print(f"\nConfigurations batch {gen_idx} (configs {start + 1}-{end}):")
            for j in range(start, end):
                item = results[j]
                params = item.get("params", {})
                mtype_j = params.get("model_type")
                model_name = MODEL_TYPE_MAP.get(mtype_j, "UNKNOWN")
                lr = params.get("learning_rate")
                bs = params.get("batch_size")
                

    # Store top unique models (like GA)
    per_model = sorted(
        [
            {"model_type": k, "params": v["params"], "fitness": v["score"]}
            for k, v in best_by_model.items()
        ],
        key=lambda x: x["fitness"],
        reverse=True
    )
    obj.top_k_results = [
        {"rank": idx + 1, "fitness": item["fitness"], "params": item["params"]}
        for idx, item in enumerate(per_model[:5])
    ]
    obj.best_params = best_params
    obj.best_score = best_score
    obj.best_model_params = best_params
    obj.best_by_model = best_by_model
    obj.best_fitness = best_score
    obj._automl_best_by_model = {
        k: {"individual": None, "fitness": v["score"]}
        for k, v in best_by_model.items()
    }

    if batch_size > 0 and n_iter % batch_size != 0:
        gen_idx = n_iter // batch_size + 1
        start = n_iter - (n_iter % batch_size)
        end = n_iter
        print(f"\nConfigurations batch {gen_idx} (configs {start + 1}-{end}):")
        for j in range(start, end):
            item = results[j]
            params = item.get("params", {})
            mtype_j = params.get("model_type")
            model_name = MODEL_TYPE_MAP.get(mtype_j, "UNKNOWN")
            lr = params.get("learning_rate")
            bs = params.get("batch_size")
            print(
                f"  - {j + 1}/{n_iter} | {model_name} | f1={item.get('mean_test_score', 0.0):.4f} | "
                f"lr={lr:.6f} | batch={bs}"
            )

    print("\n" + "=" * 50)
    print("        RANDOMIZED SEARCH RESULTS")
    print("=" * 50)
    print(f"  Best F1 Score        : {best_score:.4f}")
    print("\n  Best params per model (top 5):")
    for item in obj.top_k_results:
        mtype = item.get("params", {}).get("model_type")
        model_name = MODEL_TYPE_MAP.get(mtype, "UNKNOWN")
        print(f"  - {model_name} | f1={item.get('fitness', 0.0):.4f} | params={item.get('params')}")
    print("=" * 50)
