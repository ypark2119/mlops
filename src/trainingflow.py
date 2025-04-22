from metaflow import FlowSpec, step
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import mlflow
import mlflow.sklearn


class TrainingFlow(FlowSpec):

    @step
    def start(self):
        self.feature_sets = {
            "full": load_iris().feature_names,
            "reduced1": ['petal length (cm)', 'petal width (cm)'],
            "reduced2": ['sepal length (cm)', 'sepal width (cm)'],
            "reduced3": ['sepal length (cm)', 'petal length (cm)']
        }
        self.next(self.load_data)

    @step
    def load_data(self):
        iris = load_iris(as_frame=True)
        df = iris.frame
        self.X = df.drop(columns="target")
        self.y = df["target"]
        self.X_train_all, self.X_test_all, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42)
        self.next(self.tune_models)

    @step
    def tune_models(self):
        mlflow.set_tracking_uri("http://127.0.0.1:5050")
        mlflow.set_experiment("metaflow_experiment")

        def objective(params, X_train, y_train, feature_set_name):
            with mlflow.start_run(nested=True):
                model_type = params["type"]
                model_params = {k: v for k, v in params.items() if k != "type"}

                if model_type == 'dt':
                    clf = DecisionTreeClassifier(**model_params)
                elif model_type == 'rf':
                    clf = RandomForestClassifier(**model_params)
                elif model_type == 'xgb':
                    clf = XGBClassifier(
                        use_label_encoder=False,
                        eval_metric='mlogloss',
                        verbosity=0,
                        **model_params
                    )
                else:
                    return {'loss': 1, 'status': STATUS_OK}

                acc = cross_val_score(clf, X_train, y_train, cv=5).mean()

                mlflow.set_tag("model_type", model_type)
                mlflow.set_tag("feature_set", feature_set_name)
                mlflow.log_param("features_used", ', '.join(X_train.columns))
                mlflow.log_params(model_params)
                mlflow.log_metric("accuracy", acc)
                mlflow.sklearn.log_model(clf, artifact_path="model", input_example=X_train.iloc[:1])

                return {
                    'loss': -acc,
                    'status': STATUS_OK,
                    'params': model_params,
                    'model_type': model_type,
                    'accuracy': acc,
                    'feature_set': feature_set_name
                }

        self.search_space = hp.choice("classifier_type", [
            {
                'type': 'dt',
                'criterion': hp.choice('dtree_criterion', ['gini', 'entropy']),
                'max_depth': hp.choice('dtree_max_depth', [None, hp.randint('dtree_max_depth_int', 1, 10)]),
                'min_samples_split': hp.randint('dtree_min_samples_split', 2, 10)
            },
            {
                'type': 'rf',
                'n_estimators': hp.randint('rf_n_estimators', 20, 500),
                'max_features': hp.randint('rf_max_features', 1, 4),
                'criterion': hp.choice('rf_criterion', ['gini', 'entropy'])
            },
            {
                'type': 'xgb',
                'n_estimators': hp.randint('xgb_n_estimators', 20, 100),
                'max_depth': hp.randint('xgb_max_depth', 1, 10),
                'learning_rate': hp.uniform('xgb_learning_rate', 0.01, 0.3),
                'subsample': hp.uniform('xgb_subsample', 0.5, 1.0),
                'colsample_bytree': hp.uniform('xgb_colsample_bytree', 0.5, 1.0)
            }
        ])

        self.all_trials = []

        for feature_set_name, feature_columns in self.feature_sets.items():
            X_train = self.X_train_all[feature_columns]

            with mlflow.start_run(run_name=f"{feature_set_name}_features"):
                def wrapped(params):
                    return objective(params, X_train, self.y_train, feature_set_name)

                trials = Trials()
                fmin(fn=wrapped, space=self.search_space, algo=tpe.suggest, max_evals=20, trials=trials)
                self.all_trials.extend(trials.trials)

        self.top_3 = sorted(self.all_trials, key=lambda t: t['result']['loss'])[:3]
        self.next(self.train_top_models)

    @step
    def train_top_models(self):
        mlflow.set_tracking_uri("http://127.0.0.1:5050")
        mlflow.set_experiment("metaflow_experiment")

        self.trained_top_models = []

        for i, trial in enumerate(self.top_3, 1):
            result = trial['result']
            params = result['params']
            model_type = result['model_type']
            feature_set_name = result['feature_set']
            features = self.feature_sets[feature_set_name]
            X_train_final = self.X_train_all[features]

            if model_type == 'dt':
                model = DecisionTreeClassifier(**params)
            elif model_type == 'rf':
                model = RandomForestClassifier(**params)
            elif model_type == 'xgb':
                model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', verbosity=0, **params)

            model.fit(X_train_final, self.y_train)
            val_accuracy = cross_val_score(model, X_train_final, self.y_train, cv=5).mean()

            with mlflow.start_run(run_name=f"top_model_{i}"):
                mlflow.set_tag("model_rank", i)
                mlflow.set_tag("model_type", model_type)
                mlflow.set_tag("feature_set", feature_set_name)
                mlflow.log_params(params)
                mlflow.log_param("features_used", ', '.join(features))
                mlflow.log_metric("validation_accuracy", val_accuracy)
                mlflow.sklearn.log_model(model, artifact_path="model", input_example=X_train_final.iloc[:1],
                                         registered_model_name=f"top_model_{i}")

            self.trained_top_models.append((model, val_accuracy, model_type, features, params))

        self.next(self.register_best)

    @step
    def register_best(self):
        mlflow.set_tracking_uri("http://127.0.0.1:5050")
        mlflow.set_experiment("metaflow_experiment")

        best_model, _, model_type, features, params = self.trained_top_models[0]
        X_train_final = self.X_train_all[features]
        X_test_final = self.X_test_all[features]

        best_model.fit(X_train_final, self.y_train)
        test_accuracy = best_model.score(X_test_final, self.y_test)

        with mlflow.start_run(run_name="final_model"):
            mlflow.set_tag("final_model", model_type)
            mlflow.set_tag("feature_set", ', '.join(features))
            mlflow.log_params(params)
            mlflow.log_param("features_used", ', '.join(features))
            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.sklearn.log_model(best_model, artifact_path="final_model", input_example=X_test_final.iloc[:1],
                                     registered_model_name="iris_final_model")

        self.next(self.end)

    @step
    def end(self):
        print("Training pipeline completed and best model registered!")


if __name__ == "__main__":
    TrainingFlow()