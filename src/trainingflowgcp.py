from metaflow import FlowSpec, step, kubernetes, resources, timeout, retry, catch
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

    @kubernetes(image="us-west2-docker.pkg.dev/mlops-lab-457707/mlflow-repo/trainingflow:latest")
    @step
    def start(self):
        self.feature_sets = {
            "full": load_iris().feature_names,
            "reduced1": ['petal length (cm)', 'petal width (cm)'],
            "reduced2": ['sepal length (cm)', 'sepal width (cm)'],
            "reduced3": ['sepal length (cm)', 'petal length (cm)']
        }
        self.next(self.load_data)

    @kubernetes(image="us-west2-docker.pkg.dev/mlops-lab-457707/mlflow-repo/trainingflow:latest")
    @step
    def load_data(self):
        iris = load_iris(as_frame=True)
        df = iris.frame
        self.X = df.drop(columns="target")
        self.y = df["target"]
        self.X_train_all, self.X_test_all, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42)
        self.next(self.tune_models)

    @kubernetes(image="us-west2-docker.pkg.dev/mlops-lab-457707/mlflow-repo/trainingflow:latest")
    @resources(cpu=2, memory=16384)
    @timeout(seconds=600)
    @retry(times=2)
    @catch(var="error_info")
    @step
    def tune_models(self):
        mlflow.set_tracking_uri("https://mlflow-service-490537859363.us-west2.run.app")
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
                    clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', verbosity=0, **model_params)
                else:
                    return {'loss': 1, 'status': STATUS_OK}

                clf.fit(X_train, y_train)
                acc = clf.score(X_train, y_train)

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

        print("Environment bootstrapped. Beginning tuning...")

        self.search_space = hp.choice("classifier_type", [
            {
                'type': 'dt',
                'criterion': hp.choice('dtree_criterion', ['gini'])
            }
        ])

        self.all_trials = []

        for feature_set_name, feature_columns in self.feature_sets.items():
            print(f"Running tuning for feature set: {feature_set_name}")
            X_train = self.X_train_all[feature_columns]

            with mlflow.start_run(run_name=f"{feature_set_name}_features"):
                def wrapped(params):
                    print(f"Trying model: {params['type']} with params: {params}")
                    return objective(params, X_train, self.y_train, feature_set_name)

                trials = Trials()
                fmin(fn=wrapped, space=self.search_space, algo=tpe.suggest, max_evals=3, trials=trials)
                self.all_trials.extend(trials.trials)

        self.top_3 = sorted(self.all_trials, key=lambda t: t['result']['loss'])[:3]
        self.next(self.train_top_models)

    @kubernetes(image="us-west2-docker.pkg.dev/mlops-lab-457707/mlflow-repo/trainingflow:latest")
    @resources(cpu=2, memory=16384)
    @step
    def train_top_models(self):
        mlflow.set_tracking_uri("https://mlflow-service-490537859363.us-west2.run.app")
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

    @kubernetes(image="us-west2-docker.pkg.dev/mlops-lab-457707/mlflow-repo/trainingflow:latest")
    @step
    def register_best(self):
        mlflow.set_tracking_uri("https://mlflow-service-490537859363.us-west2.run.app")
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

    @kubernetes(image="us-west2-docker.pkg.dev/mlops-lab-457707/mlflow-repo/trainingflow:latest")
    @step
    def end(self):
        print("Training pipeline completed and best model registered!")


if __name__ == "__main__":
    TrainingFlow()