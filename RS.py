from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from MLP import get_mlp_param, create_mlp_model
from LSTM import get_lstm_param, create_lstm_model
from DNN import get_dnn_param, create_dnn_model
from CNN import get_cnn_param, create_cnn_model
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping
import itertools


def randomized_search_optimization(obj, testing_model='DNN'):

    if testing_model == 'LSTM':
    # verbose=0 suppresses Keras per-epoch training messages
        model = KerasClassifier(model=create_lstm_model, obj=obj, verbose=0)
        param_grid = get_lstm_param()
    elif testing_model == 'MLP':
        model = KerasClassifier(model=create_mlp_model, obj=obj, verbose=0)
        param_grid = get_mlp_param()
    elif testing_model == 'DNN':
        model = KerasClassifier(model=create_dnn_model, obj=obj, verbose=0)
        param_grid = get_dnn_param()
    elif testing_model == 'CNN':
        model = KerasClassifier(model=create_cnn_model, obj=obj, verbose=0)
        param_grid = get_cnn_param()
    else:
        raise ValueError(f"Unsupported model type: {testing_model}")
    # Calculate total number of combinations
    total_combinations = 1
    for values in param_grid.values():
        if isinstance(values, (list, tuple)):
            total_combinations *= len(values)

    print("=" * 50)
    print("        RANDOMIZED SEARCH CONFIGURATION")
    print("=" * 50)
    print(f"  Parameters being searched:")
    for param, values in param_grid.items():
        print(f"    - {param}: {values}")
    print(f"\n  Total possible combinations: {total_combinations}")
    print(f"  Iterations (random samples): 10")
    print(f"  Cross-validation folds     : 3")
    print(f"  Scoring metric             : recall")
    print(f"  Total fits                 : {10 * 3}  (iterations x folds)")
    print("=" * 50)
    

    search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=3, scoring='recall_weighted', n_jobs=1, verbose=0, n_iter=10)

    print("\nStarting Randomized Search...\n")

    early_stopping = EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=0
    )
    
    search.fit(obj.X_train, obj.y_train,
               validation_data=(obj.X_val, obj.y_val),
               epochs=50,
                callbacks=[early_stopping]
    )

    # Print all combinations with their scores manually
    results = search.cv_results_
    n_iter = len(results['params'])
    print("\n" + "=" * 50)
    print("      ALL SAMPLED COMBINATIONS & SCORES")
    print("=" * 50)
    for i in range(len(results['params'])):
        score = results['mean_test_score'][i]
        params = results['params'][i]
        print(f"\n  Sample {i + 1}/{n_iter}:")
        for param, value in params.items():
            print(f"    - {param}: {value}")
        print(f"    => Mean Recall: {score:.4f}")

    print("\n" + "=" * 50)
    print("        RANDOMIZED SEARCH RESULTS")
    print("=" * 50)
    print(f"  Best Recall Score    : {search.best_score_:.4f}")
    print(f"  Best Hyperparameters :")
    for param, value in search.best_params_.items():
        print(f"    - {param}: {value}")
    print("=" * 50)
