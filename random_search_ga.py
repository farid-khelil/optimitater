#!/usr/bin/env python3
"""Random search for GA hyperparameters using the existing project modules.

This file is only an orchestration script:
    - load_data.py prepares the train/validation/test data.
    - GA.py runs the genetic algorithm.
    - eval.py is used by GA.py to evaluate each individual.

It tunes GA-level hyperparameters, not neural-network architecture
hyperparameters. The sampled space is:

    population_size: 5..10
    mutation_rate: 0.005..0.03
    crossover_rate: 0.7..0.9
    generations: 5
"""

import argparse
import csv
import datetime as dt
import gc
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev

# Configure TensorFlow before importing modules that may import it.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

ON_KAGGLE = os.path.isdir("/kaggle/working")
BASE_OUT = "/kaggle/working" if ON_KAGGLE else "/home/farid/pfe"
DATA_ROOT = "/kaggle/input" if ON_KAGGLE else "/home/farid/pfe/data/processed"
DEFAULT_DATASET_IDX = "4" if ON_KAGGLE else "1"
DEFAULT_DATA_PATH = (
    f"{DATA_ROOT}/riss-dataset/RISS.csv"
    if ON_KAGGLE
    else f"{DATA_ROOT}/ransomware/RBA_small_random_search.xlsx"
)

SPACE = {
    "population_size": (50, 150),
    "mutation_rate": (0.005, 0.03),
    "crossover_rate": (0.7, 0.9),
    "generations": 5,
}


@dataclass(frozen=True)
class GAConfig:
    population_size: int
    mutation_rate: float
    crossover_rate: float
    generations: int


class ExperimentModel:
    """Small state object shared by load_data.py, GA.py, and eval.py.

    main.py defines a similar MODEL class, but importing main.py would execute
    training immediately because it has top-level run code. This object keeps
    random search reusable while still delegating real work to the existing
    load_data, GA, and eval modules.
    """

    def __init__(self, data_path):
        from sklearn.preprocessing import LabelEncoder, StandardScaler

        self.data_path = data_path

        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.X_train_val = None
        self.y_train_val = None
        self.cv_indices = None

        self.n_features = None
        self.n_classes = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.result_encoder = LabelEncoder()
        self.home_encoder = LabelEncoder()
        self.away_encoder = LabelEncoder()

        self.enable_smoothing = False
        self.smoothing_window = 9
        self.use_cv = True
        self.cv_folds = 5
        self.fixed_epochs = 100

        self.population_size = 50
        self.generations = SPACE["generations"]
        self.crossover_rate = 0.8
        self.mutation_rate = 0.01
        self.crossover_prob = self.crossover_rate
        self.mutation_prob = self.mutation_rate

        self.best_individual = None
        self.best_fitness = 0.0
        self.best_metrics = {}
        self.top_k_results = []
        self.logbook = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune GA hyperparameters using random search."
    )
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, help="Dataset path.")
    parser.add_argument(
        "--dataset-idx",
        default=DEFAULT_DATASET_IDX,
        help="Dataset selector passed to load_data() (1=RBA, 2=WPD, 3=PEHF, 4=RISS).",
    )
    parser.add_argument(
        "--test",
        default="AUTOML",
        choices=["AUTOML", "AUTO", "ALL", "UNIFIED", "MLP", "CNN", "DNN", "LSTM", "RNN"],
        help="GA search mode/model type.",
    )
    parser.add_argument("--n-configs", type=int, default=5, help="Number of GA configs.")
    parser.add_argument(
        "--runs-per-config",
        type=int,
        default=3,
        help="Repeated runs per configuration.",
    )
    parser.add_argument("--fixed-epochs", type=int, default=100)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=43)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(BASE_OUT, "results", "random_search_ga"),
        help="Directory where CSV/JSON/Markdown reports are written.",
    )
    parser.add_argument(
        "--enable-smoothing",
        action="store_true",
        help="Enable the optional feature smoothing used by load_data.py.",
    )
    parser.add_argument("--smoothing-window", type=int, default=9)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sampled configurations without running GA.",
    )
    args = parser.parse_args()

    if args.n_configs <= 0:
        parser.error("--n-configs must be positive.")
    if args.runs_per_config <= 0:
        parser.error("--runs-per-config must be positive.")
    if args.fixed_epochs <= 0:
        parser.error("--fixed-epochs must be positive.")
    if args.cv_folds <= 0:
        parser.error("--cv-folds must be positive.")

    return args


def set_global_seed(seed):
    import numpy as np
    import tensorflow as tf

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def sample_configs(args):
    rng = random.Random(args.base_seed)
    pop_low, pop_high = SPACE["population_size"]
    mut_low, mut_high = SPACE["mutation_rate"]
    cx_low, cx_high = SPACE["crossover_rate"]

    configs = []
    for _ in range(args.n_configs):
        configs.append(
            GAConfig(
                population_size=rng.randint(pop_low, pop_high),
                mutation_rate=rng.uniform(mut_low, mut_high),
                crossover_rate=rng.uniform(cx_low, cx_high),
                generations=SPACE["generations"],
            )
        )
    return configs


def build_experiment(args):
    from load_data import load_data

    obj = ExperimentModel(args.data_path)
    obj.fixed_epochs = args.fixed_epochs
    obj.cv_folds = args.cv_folds
    obj.use_cv = args.cv_folds > 1
    obj.enable_smoothing = args.enable_smoothing
    obj.smoothing_window = args.smoothing_window
    load_data(obj, idx=args.dataset_idx)
    return obj


def apply_ga_config(obj, config):
    obj.population_size = config.population_size
    obj.generations = config.generations
    obj.mutation_rate = config.mutation_rate
    obj.crossover_rate = config.crossover_rate
    obj.mutation_prob = config.mutation_rate
    obj.crossover_prob = config.crossover_rate


def json_safe(value):
    import numpy as np

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def run_config_once(config_id, run_id, config, args):
    import tensorflow as tf

    from GA import run_ga_optimization

    seed = args.base_seed + (config_id * 1000) + run_id
    set_global_seed(seed)

    obj = build_experiment(args)
    apply_ga_config(obj, config)

    started = time.time()
    error = ""
    try:
        execution_time = run_ga_optimization(obj, test=args.test, seed=seed)
        best_fitness = float(getattr(obj, "best_fitness", 0.0) or 0.0)
        best_individual = json_safe(list(obj.best_individual)) if obj.best_individual else []
        top_k_results = json_safe(getattr(obj, "top_k_results", []))
    except Exception as exc:
        execution_time = time.time() - started
        best_fitness = 0.0
        best_individual = []
        top_k_results = []
        error = str(exc)
        print(f"Run failed for config {config_id}, run {run_id}: {error}")
    finally:
        tf.keras.backend.clear_session()
        gc.collect()

    return {
        "config_id": config_id,
        "run_id": run_id,
        "seed": seed,
        **asdict(config),
        "fitness": best_fitness,
        "execution_time_seconds": float(execution_time),
        "best_individual_json": json.dumps(best_individual),
        "top_k_results_json": json.dumps(top_k_results),
        "error": error,
    }


def summarize_results(run_rows):
    grouped = {}
    for row in run_rows:
        grouped.setdefault(row["config_id"], []).append(row)

    summaries = []
    for config_id, rows in sorted(grouped.items()):
        fitness_values = [row["fitness"] for row in rows]
        execution_times = [row["execution_time_seconds"] for row in rows]
        first = rows[0]
        summaries.append(
            {
                "config_id": config_id,
                "population_size": first["population_size"],
                "mutation_rate": first["mutation_rate"],
                "crossover_rate": first["crossover_rate"],
                "generations": first["generations"],
                "runs": len(rows),
                "average_fitness": mean(fitness_values),
                "best_fitness": max(fitness_values),
                "std_fitness": stdev(fitness_values) if len(fitness_values) > 1 else 0.0,
                "total_execution_time_seconds": sum(execution_times),
                "average_execution_time_seconds": mean(execution_times),
                "failed_runs": sum(1 for row in rows if row["error"]),
            }
        )

    return sorted(
        summaries,
        key=lambda item: (item["average_fitness"], item["best_fitness"]),
        reverse=True,
    )


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_reports(output_dir, run_rows, summary_rows, args):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_csv = output_dir / "random_search_ga_runs.csv"
    summary_csv = output_dir / "random_search_ga_summary.csv"
    summary_json = output_dir / "random_search_ga_summary.json"
    report_md = output_dir / "random_search_ga_report.md"

    write_csv(runs_csv, run_rows, list(run_rows[0].keys()))
    write_csv(summary_csv, summary_rows, list(summary_rows[0].keys()))

    payload = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "space": SPACE,
        "arguments": vars(args),
        "summary": summary_rows,
        "runs": run_rows,
    }
    with open(summary_json, "w", encoding="utf-8") as file:
        json.dump(json_safe(payload), file, indent=2)

    best = summary_rows[0]
    with open(report_md, "w", encoding="utf-8") as file:
        file.write("# GA Hyperparameter Tuning Using Random Search\n\n")
        file.write(f"- Search mode: {args.test}\n")
        file.write(f"- Dataset path: `{args.data_path}`\n")
        file.write(f"- Dataset idx: `{args.dataset_idx}`\n")
        file.write(f"- Configurations sampled: {args.n_configs}\n")
        file.write(f"- Runs per configuration: {args.runs_per_config}\n")
        file.write(f"- Fixed model epochs: {args.fixed_epochs}\n\n")
        file.write("## Search Space\n\n")
        for key, value in SPACE.items():
            file.write(f"- {key}: {value}\n")
        file.write("\n## Best Configuration\n\n")
        file.write(f"- Population size: {best['population_size']}\n")
        file.write(f"- Mutation rate: {best['mutation_rate']:.5f}\n")
        file.write(f"- Crossover rate: {best['crossover_rate']:.5f}\n")
        file.write(f"- Generations: {best['generations']}\n")
        file.write(f"- Average fitness: {best['average_fitness']:.4f}\n")
        file.write(f"- Best fitness: {best['best_fitness']:.4f}\n")
        file.write(f"- Fitness std: {best['std_fitness']:.4f}\n\n")
        file.write("## Ranked Configurations\n\n")
        file.write(
            "| Rank | Config | Pop | Mut | Cross | Gen | Avg Fitness | Best Fitness | Std |\n"
        )
        file.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for rank, row in enumerate(summary_rows, start=1):
            file.write(
                f"| {rank} | {row['config_id']} | {row['population_size']} | "
                f"{row['mutation_rate']:.5f} | {row['crossover_rate']:.5f} | "
                f"{row['generations']} | "
                f"{row['average_fitness']:.4f} | {row['best_fitness']:.4f} | "
                f"{row['std_fitness']:.4f} |\n"
            )

    return runs_csv, summary_csv, summary_json, report_md


def main():
    args = parse_args()
    configs = sample_configs(args)

    print("Random Search GA configuration")
    print(f"Data path: {args.data_path}")
    print(f"Dataset idx: {args.dataset_idx}")
    print(f"Search mode: {args.test}")
    print(f"Search space: {SPACE}")
    print(f"Sampled configurations: {len(configs)}")
    print(f"Runs per configuration: {args.runs_per_config}")

    for config_id, config in enumerate(configs, start=1):
        print(f"Config {config_id}: {asdict(config)}")

    if args.dry_run:
        print("Dry run requested; no GA runs were executed.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_rows = []
    total_runs = len(configs) * args.runs_per_config
    run_counter = 0
    for config_id, config in enumerate(configs, start=1):
        for run_id in range(1, args.runs_per_config + 1):
            run_counter += 1
            print(
                f"\nStarting run {run_counter}/{total_runs} "
                f"(config={config_id}, repeat={run_id})"
            )
            row = run_config_once(config_id, run_id, config, args)
            run_rows.append(row)

            partial_summary = summarize_results(run_rows)
            write_reports(output_dir, run_rows, partial_summary, args)
            print(
                f"Finished run {run_counter}/{total_runs}: "
                f"fitness={row['fitness']:.4f}, "
                f"time={row['execution_time_seconds']:.2f}s"
            )

    summary_rows = summarize_results(run_rows)
    paths = write_reports(output_dir, run_rows, summary_rows, args)
    best = summary_rows[0]

    print("\nRandom search finished.")
    print(f"Best config id: {best['config_id']}")
    print(f"Average fitness: {best['average_fitness']:.4f}")
    print(f"Best fitness: {best['best_fitness']:.4f}")
    print(f"Reports written to: {output_dir}")
    for path in paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
