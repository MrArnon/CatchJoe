import pandas as pd
import numpy as np
import json
import pickle
import logging
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
import preprocessing as prep


def load_config(config_name='config.json'):
    path_config = Path(config_name)
    with(open(path_config, 'r')) as file:
        config = json.load(file)
    return config


def calc_val_scores(X, Y, config):
    cv = StratifiedKFold(n_splits=config['features']['folds'], random_state=1, shuffle=True)

    model = DecisionTreeClassifier()

    scores = cross_validate(model, X, Y, scoring=['roc_auc', 'f1', 'accuracy', 'recall', 'precision'],
                            cv=cv, n_jobs=-1, return_train_score=True)
    for key in scores.keys():
        scores[key] = scores[key].tolist()

    Path(config['metrics_path']).mkdir(parents=True, exist_ok=True)
    with open(Path(config['metrics_path']) / Path('metrics.json'), 'w') as fp:
        json.dump(scores, fp, sort_keys=True, indent=4, separators=(',', ':'), )
        logging.info(f"Validation metrics saved {Path(config['metrics_path'])/Path('metrics.json')}")

    return cv


def train_trees_by_folds(X, Y, cv):
    models = []
    for train, test in cv.split(X, Y):
        x_train, y_train = X.iloc[train], Y.iloc[train]
        model = DecisionTreeClassifier()
        models.append(model.fit(x_train, y_train))

    return models


def predict_test_data(models, config, return_result=False):
    path = config['output']['test']
    df_test = pd.read_csv(Path(config['data_path']) / Path(path), index_col=0)

    df = prep.read_df(config['data_path'], config['input']['test'])
    Path(config['model_path']).mkdir(parents=True, exist_ok=True)
    preds = []
    importances = []
    i = 0
    for model in models:
        pred = model.predict(df_test)
        with open(Path(config['model_path']) / Path(f'decision_tree_fold_{i}.pkl'), 'wb') as f:
            pickle.dump(model, f)
            logging.info(f"Model saved {Path(config['model_path'])/Path(f'decision_tree_fold_{i}.pkl')}")
            i += 1
        preds.append(pred)
        importances.append(np.ndarray.round(model.feature_importances_, 4))

    df_importances = pd.DataFrame(importances, columns=df_test.columns)
    final_pred = np.mean(preds, axis=0).astype(int)
    df_test['predictions'] = final_pred
    df_final = pd.DataFrame(df_test['predictions'].groupby(df_test.index).mean().astype(int))

    df = df.join(df_final)

    path_csv = Path(config['data_path']) / Path(config['output']['prediction'])
    path_json = Path(config['data_path']) / Path(config['output']['prediction'].replace('csv', 'json'))
    path_importances = Path(config['metrics_path']) / Path('feature_importances.json')
    df.to_csv(path_csv)
    logging.info(f"Predicted results saved {path_csv}")
    df.to_json(path_json)
    logging.info(f"Predicted results saved {path_json}")
    df_importances.to_json(path_importances)
    logging.info(f"Feature importances saved {path_importances}")
    if return_result:
        return df


def main():

    logging.info("Start preprocessing data")
    prep.main()
    logging.info("End preprocessing data")

    config = load_config()

    path = config['output']['train']
    target_col = config['features']['target_col']

    df = pd.read_csv(Path(config['data_path']) / Path(path), index_col=0)
    X = df.drop(target_col, axis=1)
    Y = df[[target_col]]
    cv = calc_val_scores(X, Y, config)
    models = train_trees_by_folds(X, Y, cv)
    predict_test_data(models, config)


if __name__ == '__main__':
    config = load_config()
    Path(config['log_path']).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=Path(config['log_path']) / Path('pipeline.log'),
                        level=logging.DEBUG, format='%(asctime)s %(message)s')
    main()
