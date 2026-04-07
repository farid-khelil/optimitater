from mealpy.swarm_based import GWO
import numpy as np
from MLP import create_mlp_model
from LSTM import create_lstm_model
from CNN import create_cnn_model
from DNN import create_dnn_model
from RNN import create_rnn_model
from mealpy.utils.problem import Problem
from mealpy.utils.space import IntegerVar, FloatVar, CategoricalVar
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import gc


def _print_section(title):
    line = "═" * 72
    print(f"\n{line}")
    print(f"  {title}")
    print(line)


def _print_kv(label, value):
    print(f"  • {label:<16}: {value}")


def _to_scalar(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.array(value).flatten()
        if arr.size == 0:
            return None
        return arr[0]
    return value


def _decode_choice(raw, choices):
    raw = _to_scalar(raw)

    if raw in choices:
        return raw

    try:
        v = float(raw)
    except Exception:
        return choices[0]

    idx = int(round(v))
    # 0-based index
    if 0 <= idx < len(choices):
        return choices[idx]
    # 1-based index
    if 1 <= idx <= len(choices):
        return choices[idx - 1]

    # fallback for numeric choices: nearest value
    if all(isinstance(c, (int, float, np.integer, np.floating)) for c in choices):
        return min(choices, key=lambda c: abs(float(c) - v))

    return choices[0]


def _decode_solution(solution, test='MLP'):
    solution = np.array(solution, dtype=object).flatten()

    activation_decode = ['relu', 'elu', 'selu', 'tanh']
    lr_decode = [0.0001, 0.001, 0.005, 0.01, 0.05]

    if test == 'MLP':
        n_layers = int(round(float(_to_scalar(solution[0]))))
        n_layers = max(1, min(5, n_layers))
        dense_units = [int(_decode_choice(solution[i], [16, 32, 64, 128, 256])) for i in range(1, 6)][:n_layers]
        return {
            "n_layers": n_layers,
            "dense_units": dense_units,
            "dropout": max(0.0, min(0.5, float(_to_scalar(solution[6])))),
            "learning_rate": float(_decode_choice(solution[7], lr_decode)),
            "optimizer_idx": int(_decode_choice(solution[8], [0, 1, 2])),
            "activation": _decode_choice(solution[9], activation_decode),
            "batch_size": int(_decode_choice(solution[10], [16, 32, 64, 128])),
        }

    if test == 'LSTM':
        n_lstm_layers = int(round(float(_to_scalar(solution[0]))))
        n_lstm_layers = max(1, min(3, n_lstm_layers))
        lstm_units = [int(_decode_choice(solution[i], [32, 64, 128])) for i in range(1, 4)][:n_lstm_layers]
        n_dense_layers = int(round(float(_to_scalar(solution[6]))))
        n_dense_layers = max(1, min(3, n_dense_layers))
        dense_units = [int(_decode_choice(solution[i], [64, 128, 256])) for i in range(7, 10)][:n_dense_layers]
        return {
            "n_lstm_layers": n_lstm_layers,
            "lstm_units": lstm_units,
            "dropout_rate": float(_decode_choice(solution[4], [0.0, 0.1, 0.2, 0.3, 0.5])),
            "rec_dropout_rate": float(_decode_choice(solution[5], [0.0, 0.1, 0.2])),
            "n_dense_layers": n_dense_layers,
            "dense_units": dense_units,
            "learning_rate": float(_decode_choice(solution[10], lr_decode)),
            "optimizer_idx": int(_decode_choice(solution[11], [0, 1, 2])),
            "activation": _decode_choice(solution[12], activation_decode),
            "batch_size": int(_decode_choice(solution[13], [16, 32, 64])),
        }

    if test == 'CNN':
        n_conv_layers = int(round(float(_to_scalar(solution[0]))))
        n_conv_layers = max(1, min(3, n_conv_layers))
        conv_filters = [int(_decode_choice(solution[i], [32, 64, 128])) for i in range(1, 4)][:n_conv_layers]
        kernel_sizes = [int(_decode_choice(solution[i], [1, 2, 3])) for i in range(4, 7)][:n_conv_layers]
        pool_sizes = [int(_decode_choice(solution[i], [2, 4, 8])) for i in range(7, 10)][:n_conv_layers]
        n_dense_layers = int(round(float(_to_scalar(solution[10]))))
        n_dense_layers = max(1, min(5, n_dense_layers))
        dense_units = [int(_decode_choice(solution[i], [64, 128, 256, 512])) for i in range(11, 16)][:n_dense_layers]
        return {
            "n_conv_layers": n_conv_layers,
            "conv_filters": conv_filters,
            "kernel_sizes": kernel_sizes,
            "pool_sizes": pool_sizes,
            "n_dense_layers": n_dense_layers,
            "dense_units": dense_units,
            "dropout_rate": float(_decode_choice(solution[16], [0.0, 0.1, 0.2, 0.3, 0.5])),
            "learning_rate": float(_decode_choice(solution[17], lr_decode)),
            "optimizer_idx": int(_decode_choice(solution[18], [0, 1, 2])),
            "activation": _decode_choice(solution[19], activation_decode),
            "batch_size": int(_decode_choice(solution[20], [16, 32, 64, 128])),
        }

    if test == 'RNN':
        n_layers = int(round(float(_to_scalar(solution[0]))))
        n_layers = max(1, min(3, n_layers))
        n_dense = int(round(float(_to_scalar(solution[2]))))
        n_dense = max(1, min(5, n_dense))
        dense_units = [int(_decode_choice(solution[i], [64, 128, 256, 512])) for i in range(3, 8)][:n_dense]
        return {
            "n_layers": n_layers,
            "rnn_units": int(_decode_choice(solution[1], [64, 128, 256])),
            "n_dense": n_dense,
            "dense_units": dense_units,
            "dropout_rate": float(_decode_choice(solution[8], [0.0, 0.1, 0.2, 0.3, 0.5])),
            "learning_rate": float(_decode_choice(solution[9], lr_decode)),
            "optimizer_idx": int(_decode_choice(solution[10], [0, 1, 2])),
            "activation": _decode_choice(solution[11], activation_decode),
            "batch_size": int(_decode_choice(solution[12], [16, 32, 64, 128])),
        }

    # DNN
    n_hidden_layers = int(round(float(_to_scalar(solution[0]))))
    n_hidden_layers = max(1, min(3, n_hidden_layers))
    first_hidden = int(_decode_choice(solution[1], [32, 64, 128]))
    other_hidden = int(_decode_choice(solution[5], [64, 128, 256]))
    hidden_units = [first_hidden] + [other_hidden] * max(0, n_hidden_layers - 1)
    return {
        "n_hidden_layers": n_hidden_layers,
        "hidden_units": hidden_units,
        "dropout_rate": float(_decode_choice(solution[2], [0.0, 0.1, 0.2, 0.3, 0.5])),
        "learning_rate": float(_decode_choice(solution[6], lr_decode)),
        "optimizer_idx": int(_decode_choice(solution[7], [0, 1, 2])),
        "activation": _decode_choice(solution[8], activation_decode),
        "batch_size": int(_decode_choice(solution[9], [16, 32, 64])),
    }


def _make_model(obj, decoded, test='MLP'):
    if test == 'MLP':
        return create_mlp_model(
            obj=obj,
            n_dense_layers=decoded["n_layers"],
            dense_units=decoded["dense_units"],
            dropout_rate=decoded["dropout"],
            learning_rate=decoded["learning_rate"],
            optimizer_idx=decoded["optimizer_idx"],
            activation=decoded["activation"],
        )
    if test == 'LSTM':
        
        return create_lstm_model(
            obj=obj,
            n_lstm_layers=decoded["n_lstm_layers"],
            lstm_units=decoded["lstm_units"],
            dropout_rate=decoded["dropout_rate"],
            rec_dropout_rate=decoded["rec_dropout_rate"],
            n_dense_layers=decoded["n_dense_layers"],
            dense_units=decoded["dense_units"],
            learning_rate=decoded["learning_rate"],
            optimizer_idx=decoded["optimizer_idx"],
            activation=decoded["activation"],
        )
    if test == 'CNN':
        return create_cnn_model(
            obj=obj,
            n_conv_layers=decoded["n_conv_layers"],
            conv_filters=decoded["conv_filters"],
            kernel_sizes=decoded["kernel_sizes"],
            pool_sizes=decoded["pool_sizes"],
            n_dense_layers=decoded["n_dense_layers"],
            dense_units=decoded["dense_units"],
            dropout_rate=decoded["dropout_rate"],
            learning_rate=decoded["learning_rate"],
            optimizer_idx=decoded["optimizer_idx"],
            activation=decoded["activation"],
        )
    if test == 'RNN':
        return create_rnn_model(
            obj=obj,
            n_layers=decoded["n_layers"],
            rnn_units=decoded["rnn_units"],
            n_dense=decoded["n_dense"],
            dens=decoded["dense_units"],
            optimizer_idx=decoded["optimizer_idx"],
            activation=decoded["activation"],
            dropout_rate=decoded["dropout_rate"],
            learning_rate=decoded["learning_rate"],
        )
    return create_dnn_model(
        obj=obj,
        n_hidden_layers=decoded["n_hidden_layers"],
        hidden_units=decoded["hidden_units"],
        dropout_rate=decoded["dropout_rate"],
        learning_rate=decoded["learning_rate"],
        optimizer_idx=decoded["optimizer_idx"],
        activation=decoded["activation"],
    )


def _format_best_individual(decoded, test='MLP'):
    if test == 'MLP':
        padded_units = decoded["dense_units"] + [16] * (5 - len(decoded["dense_units"]))
        return [
            decoded["n_layers"],
            *padded_units[:5],
            decoded["dropout"],
            decoded["learning_rate"],
            decoded["optimizer_idx"],
            decoded["activation"],
            decoded["batch_size"],
        ]
    if test == 'LSTM':
        lstm_pad = decoded["lstm_units"] + [64] * (3 - len(decoded["lstm_units"]))
        dense_pad = decoded["dense_units"] + [64] * (3 - len(decoded["dense_units"]))
        return [
            decoded["n_lstm_layers"], *lstm_pad[:3], decoded["dropout_rate"], decoded["rec_dropout_rate"],
            decoded["n_dense_layers"], *dense_pad[:3], decoded["learning_rate"], decoded["optimizer_idx"],
            decoded["activation"], decoded["batch_size"], 100
        ]
    if test == 'CNN':
        conv_pad = decoded["conv_filters"] + [32] * (3 - len(decoded["conv_filters"]))
        ker_pad = decoded["kernel_sizes"] + [1] * (3 - len(decoded["kernel_sizes"]))
        pool_pad = decoded["pool_sizes"] + [2] * (3 - len(decoded["pool_sizes"]))
        dense_pad = decoded["dense_units"] + [64] * (5 - len(decoded["dense_units"]))
        return [
            decoded["n_conv_layers"], *conv_pad[:3], *ker_pad[:3], *pool_pad[:3], decoded["n_dense_layers"],
            *dense_pad[:5], decoded["dropout_rate"], decoded["learning_rate"], decoded["optimizer_idx"],
            decoded["activation"], decoded["batch_size"], 100
        ]
    if test == 'RNN':
        dense_pad = decoded["dense_units"] + [64] * (5 - len(decoded["dense_units"]))
        return [
            decoded["n_layers"], decoded["rnn_units"], decoded["n_dense"], *dense_pad[:5], decoded["optimizer_idx"],
            decoded["activation"], decoded["dropout_rate"], decoded["learning_rate"], decoded["batch_size"], 100
        ]
    # DNN
    hidden_pad = decoded["hidden_units"] + [64] * (5 - len(decoded["hidden_units"]))
    return [
        decoded["n_hidden_layers"], *hidden_pad[:5], decoded["dropout_rate"], decoded["learning_rate"],
        decoded["optimizer_idx"], decoded["activation"], decoded["batch_size"], 100
    ]

def GrayWolfOptimizer(obj,test='MLP', target_evaluations=500, pop_size=15):
    if test == 'LSTM' or test == 'RNN':
        obj.X_train = obj.X_train.reshape(obj.X_train.shape[0], 1, obj.X_train.shape[1])
        obj.X_val = obj.X_val.reshape(obj.X_val.shape[0], 1, obj.X_val.shape[1])
        obj.X_test = obj.X_test.reshape(obj.X_test.shape[0], 1, obj.X_test.shape[1])
    if not hasattr(obj, "gwo_tested_solutions"):
        obj.gwo_tested_solutions = []
    else:
        obj.gwo_tested_solutions.clear()
    obj.gwo_population_bests = []

    pop_size = max(2, int(pop_size))
    target_evaluations = max(pop_size, int(target_evaluations))
    # mealpy usually evaluates initial population + each epoch population:
    # total_evals ~= (epoch + 1) * pop_size
    epoch = max(1, int(np.ceil(target_evaluations / pop_size)) - 1)

    obj.gwo_config = {
        "target_evaluations": target_evaluations,
        "pop_size": pop_size,
        "epoch": epoch,
        "estimated_evaluations": (epoch + 1) * pop_size,
    }
    eval_counter = {"n": 0}

    def fitness(solution):
        eval_counter["n"] += 1
        decoded = _decode_solution(solution, test=test)
        batch_size = decoded["batch_size"]

        model = None
        history = None
        x_train_fit, x_val_fit = obj.X_train, obj.X_val
        val_acc = 0.0

        try:
            # Clear old graph/tensors before building a new candidate model
            tf.keras.backend.clear_session()

            model = _make_model(obj=obj, decoded=decoded, test=test)

            if test in ['RNN', 'LSTM'] and len(obj.X_train.shape) == 2:
                x_train_fit = obj.X_train.reshape((-1, 1, obj.n_features))
                x_val_fit = obj.X_val.reshape((-1, 1, obj.n_features))
            elif test == 'CNN' and len(obj.X_train.shape) == 2:
                x_train_fit = obj.X_train.reshape((-1, obj.n_features, 1))
                x_val_fit = obj.X_val.reshape((-1, obj.n_features, 1))

            # --- Train with fixed epochs and early stopping ---
            early_stopping = EarlyStopping(
                    monitor='val_accuracy',
                    patience=3,
                    restore_best_weights=True,
                    verbose=0
            )

            history = model.fit(
                x_train_fit, obj.y_train,
                validation_data=(x_val_fit, obj.y_val),
                epochs=100,
                batch_size=int(batch_size),
                callbacks=[early_stopping],
                verbose=0
            )

            # --- Fitness: maximize val_accuracy, GWO minimizes → return negative ---
            val_acc = float(max(history.history.get('val_accuracy', [0.0])))
        except Exception as e:
            # OOM or training failure: penalize candidate and continue optimization
            obj.gwo_last_error = str(e)
            val_acc = 0.0
        finally:
            # Aggressive cleanup to avoid memory accumulation across evaluations
            try:
                del history
            except Exception:
                pass
            try:
                del model
            except Exception:
                pass
            gc.collect()
            tf.keras.backend.clear_session()

        obj.gwo_tested_solutions.append({
            "evaluation": eval_counter["n"],
            "params": decoded,
            "val_accuracy": float(val_acc),
            "fitness": float(-val_acc),
        })

        # Print only best solution of each population chunk
        if eval_counter["n"] % pop_size == 0:
            pop_index = eval_counter["n"] // pop_size
            chunk = obj.gwo_tested_solutions[-pop_size:]
            best_chunk = max(chunk, key=lambda x: x["val_accuracy"])
            obj.gwo_population_bests.append({
                "population": pop_index,
                "evaluation_range": (eval_counter["n"] - pop_size + 1, eval_counter["n"]),
                "best": best_chunk,
            })

            p = best_chunk["params"]
            _print_section(f"🐺 Population {pop_index} Best")
            _print_kv("Eval range", f"{eval_counter['n'] - pop_size + 1}-{eval_counter['n']}")
            _print_kv("Best val_acc", f"{best_chunk['val_accuracy']:.4f}")
            for key, value in p.items():
                _print_kv(key, value)

        return -val_acc
    if test == 'MLP':
        problem = Problem(
            obj_func=fitness,
            bounds=[
        IntegerVar(1, 5),

        CategoricalVar([16, 32, 64, 128, 256]),
        CategoricalVar([16, 32, 64, 128, 256]),
        CategoricalVar([16, 32, 64, 128, 256]),
        CategoricalVar([16, 32, 64, 128, 256]),
        CategoricalVar([16, 32, 64, 128, 256]),

        FloatVar(0.1, 0.5),

        CategoricalVar([0.0001, 0.001, 0.005, 0.01, 0.05]),

        CategoricalVar([0, 1, 2]),

        CategoricalVar(['relu', 'elu', 'selu', 'tanh']),
        CategoricalVar([16, 32, 64, 128])
        ],  
            minmax="min"
        )
    elif test == 'LSTM':
        problem = Problem(
            obj_func=fitness,
            bounds=[
            IntegerVar(1, 3),  # n_lstm_layers

            CategoricalVar([32, 64, 128]),  # lstm_units layer 1
            CategoricalVar([32, 64, 128]),  # lstm_units layer 2
            CategoricalVar([32, 64, 128]),  # lstm_units layer 3

            CategoricalVar([0.0, 0.1, 0.2, 0.3, 0.5]),  # dropout_rate
            CategoricalVar([0.0, 0.1, 0.2]),            # rec_dropout_rate

            IntegerVar(1, 3),  # n_dense_layers
            CategoricalVar([64, 128, 256]),  # dense_units layer 1
            CategoricalVar([64, 128, 256]),  # dense_units layer 2
            CategoricalVar([64, 128, 256]),  # dense_units layer 3

            CategoricalVar([0.0001, 0.001, 0.005, 0.01, 0.05]),  # learning_rate
            CategoricalVar([0, 1, 2]),                            # optimizer_idx
            CategoricalVar(['relu', 'elu', 'selu', 'tanh']),      # activation
            CategoricalVar([16, 32, 64]),                         # batch_size
           
            ],
            minmax="min"
        )
    elif test == 'CNN':
        problem = Problem(
            obj_func=fitness,
            bounds=[
            IntegerVar(1, 3),  # n_conv_layers

            CategoricalVar([32, 64, 128]),  # conv_filters layer 1
            CategoricalVar([32, 64, 128]),  # conv_filters layer 2
            CategoricalVar([32, 64, 128]),  # conv_filters layer 3

            CategoricalVar([1, 2, 3]),      # kernel_sizes layer 1
            CategoricalVar([1, 2, 3]),      # kernel_sizes layer 2
            CategoricalVar([1, 2, 3]),      # kernel_sizes layer 3

            CategoricalVar([2, 4, 8]),      # pool_sizes layer 1
            CategoricalVar([2, 4, 8]),      # pool_sizes layer 2
            CategoricalVar([2, 4, 8]),      # pool_sizes layer 3

            IntegerVar(1, 5),               # n_dense_layers
            CategoricalVar([64, 128, 256, 512]),  # dense_units layer 1
            CategoricalVar([64, 128, 256, 512]),  # dense_units layer 2
            CategoricalVar([64, 128, 256, 512]),  # dense_units layer 3
            CategoricalVar([64, 128, 256, 512]),  # dense_units layer 4
            CategoricalVar([64, 128, 256, 512]),  # dense_units layer 5

            CategoricalVar([0.0, 0.1, 0.2, 0.3, 0.5]),          # dropout_rate
            CategoricalVar([0.0001, 0.001, 0.005, 0.01, 0.05]), # learning_rate
            IntegerVar(0, 2),                                    # optimizer
            CategoricalVar(['relu', 'elu', 'selu', 'tanh']),     # activation
            CategoricalVar([16, 32, 64, 128]),                   # batch_size
            ],
            minmax="min"
        )
    elif test == 'RNN':
        problem = Problem(
            obj_func=fitness,
            bounds=[
            IntegerVar(1, 3),  # n_rnn_layers

            CategoricalVar([64, 128, 256]),  # rnn_units

            IntegerVar(1, 5),  # n_dense_layers
            CategoricalVar([64, 128, 256, 512]),  # dense_units layer 1
            CategoricalVar([64, 128, 256, 512]),  # dense_units layer 2
            CategoricalVar([64, 128, 256, 512]),  # dense_units layer 3
            CategoricalVar([64, 128, 256, 512]),  # dense_units layer 4
            CategoricalVar([64, 128, 256, 512]),  # dense_units layer 5

            CategoricalVar([0.0, 0.1, 0.2, 0.3, 0.5]),           # dropout_rate
            CategoricalVar([0.0001, 0.001, 0.005, 0.01, 0.05]),  # learning_rate
            IntegerVar(0, 2),                                     # optimizer
            CategoricalVar(['relu', 'elu', 'selu', 'tanh']),      # activation
            CategoricalVar([16, 32, 64, 128]),                    # batch_size
            ],
            minmax="min"
        )
    elif test == 'DNN':
        problem = Problem(
            obj_func=fitness,
            bounds=[
            IntegerVar(1, 3),                                # n_lstm_layers
            CategoricalVar([32, 64, 128]),                  # lstm_units
            CategoricalVar([0.0, 0.1, 0.2, 0.3, 0.5]),      # dropout_rate
            CategoricalVar([0.0, 0.1, 0.2]),                # rec_dropout_rate
            IntegerVar(1, 3),                                # n_dense_layers
            CategoricalVar([64, 128, 256]),                 # dense_units
            CategoricalVar([0.0001, 0.001, 0.005, 0.01, 0.05]),  # learning_rate
            IntegerVar(0, 2),                                # optimizer_idx
            CategoricalVar(['relu', 'elu', 'selu', 'tanh']),# activation
            CategoricalVar([16, 32, 64]),                   # batch_size
            ],
            minmax="min"
        )
    model = GWO.OriginalGWO(epoch=epoch, pop_size=pop_size)

    result = model.solve(problem)

    # mealpy compatibility:
    # - old versions may return (best_pos, best_fit)
    # - newer versions return an Agent object
    if isinstance(result, tuple) and len(result) == 2:
        best_pos, best_fit = result
    else:
        best_agent = result
        best_pos = getattr(best_agent, "solution", None)

        if hasattr(best_agent, "target") and hasattr(best_agent.target, "fitness"):
            best_fit = best_agent.target.fitness
        elif hasattr(best_agent, "fitness"):
            best_fit = best_agent.fitness
        else:
            raise TypeError("Unable to read optimization result from mealpy solve().")

    # Save best hyperparameters to obj (compatible with print_resault.decode_individual)
    decoded_best = _decode_solution(best_pos, test=test)
    obj.best_individual = _format_best_individual(decoded_best, test=test)
    obj.best_params = {"model": test, **decoded_best}
    obj.best_fitness = float(-best_fit)

    # Save alpha / beta / delta wolves from final population if available
    obj.gwo_alpha = {
        "solution": best_pos,
        "decoded": decoded_best,
        "fitness": float(best_fit),
        "score": float(-best_fit),
    }
    obj.gwo_beta = None
    obj.gwo_delta = None

    if hasattr(model, "pop") and model.pop is not None:
        try:
            ranked = sorted(
                model.pop,
                key=lambda a: a.target.fitness if hasattr(a, "target") and hasattr(a.target, "fitness") else getattr(a, "fitness", float("inf"))
            )
            if len(ranked) > 1:
                beta_agent = ranked[1]
                beta_solution = getattr(beta_agent, "solution", None)
                beta_fit = beta_agent.target.fitness if hasattr(beta_agent, "target") and hasattr(beta_agent.target, "fitness") else getattr(beta_agent, "fitness", None)
                obj.gwo_beta = {
                    "solution": beta_solution,
                    "decoded": _decode_solution(beta_solution, test=test) if beta_solution is not None else None,
                    "fitness": float(beta_fit) if beta_fit is not None else None,
                    "score": float(-beta_fit) if beta_fit is not None else None,
                }
            if len(ranked) > 2:
                delta_agent = ranked[2]
                delta_solution = getattr(delta_agent, "solution", None)
                delta_fit = delta_agent.target.fitness if hasattr(delta_agent, "target") and hasattr(delta_agent.target, "fitness") else getattr(delta_agent, "fitness", None)
                obj.gwo_delta = {
                    "solution": delta_solution,
                    "decoded": _decode_solution(delta_solution, test=test) if delta_solution is not None else None,
                    "fitness": float(delta_fit) if delta_fit is not None else None,
                    "score": float(-delta_fit) if delta_fit is not None else None,
                }
        except Exception:
            pass

    # Only register best params/fitness here.
    # Final training + test evaluation should be run separately with evaluate_best_model(...)
    obj.best_metrics = {}
    obj.model = None