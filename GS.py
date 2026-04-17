from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from MLP import get_mlp_param, create_mlp_model
from scikeras.wrappers import KerasClassifier
import itertools


def grid_search_optimization(obj):
    param_grid = get_mlp_param()

    # Calculate total number of combinations
    total_combinations = 1
    for values in param_grid.values():
        if isinstance(values, (list, tuple)):
            total_combinations *= len(values)

    print("=" * 50)
    print("           GRID SEARCH CONFIGURATION")
    print("=" * 50)
    print(f"  Parameters being searched:")
    for param, values in param_grid.items():
        print(f"    - {param}: {values}")
    print(f"\n  Total combinations   : {total_combinations}")
    print(f"  Cross-validation folds: 3")
    print(f"  Scoring metric       : recall")
    print(f"  Total fits           : {total_combinations * 3}  (combinations x folds)")
    print("=" * 50)

    # verbose=0 suppresses Keras per-epoch training messages
    model = KerasClassifier(model=create_mlp_model, obj=obj, verbose=0)

    search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='recall_weighted', n_jobs=1, verbose=0)

    print("\nStarting Grid Search...\n")
    search.fit(obj.X_train, obj.y_train,
               validation_data=(obj.X_val, obj.y_val),
               epoch=50
    )

    # Print all combinations with their scores manually
    print("\n" + "=" * 50)
    print("         ALL COMBINATIONS & SCORES")
    print("=" * 50)
    results = search.cv_results_
    for i in range(len(results['params'])):
        score = results['mean_test_score'][i]
        params = results['params'][i]
        print(f"\n  Combination {i + 1}/{total_combinations}:")
        for param, value in params.items():
            print(f"    - {param}: {value}")
        print(f"    => Mean Recall: {score:.4f}")

    print("\n" + "=" * 50)
    print("           GRID SEARCH RESULTS")
    print("=" * 50)
    print(f"  Best Recall Score    : {search.best_score_:.4f}")
    print(f"  Best Hyperparameters :")
    for param, value in search.best_params_.items():
        print(f"    - {param}: {value}")
    print("=" * 50)
