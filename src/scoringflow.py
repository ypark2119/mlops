from metaflow import FlowSpec, step
from sklearn.datasets import load_iris
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


class ScoringFlow(FlowSpec):

    @step
    def start(self):
        mlflow.set_tracking_uri("http://127.0.0.1:5050")

        iris = load_iris()
        self.X_new = iris.data[:10]

        client = MlflowClient()
        registered_model_name = "iris_final_model"

        latest_versions = client.get_latest_versions(registered_model_name)
        best_run = None
        best_accuracy = -1.0

        for version in latest_versions:
            run_id = version.run_id
            run = client.get_run(run_id)
            accuracy_metric = run.data.metrics.get("test_accuracy")
            if accuracy_metric is not None:
                test_accuracy = float(accuracy_metric)
                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy
                    best_run = run

        if best_run:
            self.model_uri = f"runs:/{best_run.info.run_id}/final_model"
            print(f"Best model found: {self.model_uri} with accuracy {best_accuracy:.4f}")
        else:
            raise ValueError("No suitable model run with test_accuracy found.")

        self.next(self.load_model)

    @step
    def load_model(self):
        self.model = mlflow.sklearn.load_model(self.model_uri)
        self.next(self.predict)

    @step
    def predict(self):
        self.predictions = self.model.predict(self.X_new)
        print("Predictions on new data:", self.predictions)
        self.next(self.end)

    @step
    def end(self):
        print("Scoring flow complete.")


if __name__ == '__main__':
    ScoringFlow()