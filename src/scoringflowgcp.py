from metaflow import FlowSpec, step, kubernetes, resources
from sklearn.datasets import load_iris
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


class ScoringFlow(FlowSpec):

    @kubernetes(
        image="us-west2-docker.pkg.dev/mlops-lab-457707/mlflow-repo/trainingflow:latest",
        service_account="argo"
    )
    @resources(cpu=1, memory=4096)
    @step
    def start(self):
        mlflow.set_tracking_uri("https://mlflow-service-490537859363.us-west2.run.app")

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
            test_accuracy = run.data.metrics.get("test_accuracy")
            if test_accuracy is not None and test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_run = run

        if best_run:
            self.run_id = best_run.info.run_id
            print(f"Best model found: {self.run_id} with accuracy {best_accuracy:.4f}")
        else:
            raise ValueError("No suitable model run with 'test_accuracy' found.")

        self.next(self.load_model)

    @kubernetes(
        image="us-west2-docker.pkg.dev/mlops-lab-457707/mlflow-repo/trainingflow:latest",
        service_account="argo"
    )
    @resources(cpu=1, memory=4096)
    @step
    def load_model(self):
        mlflow.set_tracking_uri("https://mlflow-service-490537859363.us-west2.run.app")

        client = MlflowClient()
        run = client.get_run(self.run_id)
        artifact_uri = run.info.artifact_uri
        model_path = f"{artifact_uri}/final_model"

        print(f"Loading model from: {model_path}")
        try:
            self.model = mlflow.sklearn.load_model(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")

        self.next(self.predict)

    @kubernetes(
        image="us-west2-docker.pkg.dev/mlops-lab-457707/mlflow-repo/trainingflow:latest",
        service_account="argo"
    )
    @resources(cpu=1, memory=4096)
    @step
    def predict(self):
        try:
            self.predictions = self.model.predict(self.X_new)
            print("Predictions on new data:", self.predictions.tolist())
        except Exception as e:
            raise RuntimeError(f"Model prediction failed: {e}")

        self.next(self.end)

    @kubernetes(
        image="us-west2-docker.pkg.dev/mlops-lab-457707/mlflow-repo/trainingflow:latest",
        service_account="argo"
    )
    @resources(cpu=1, memory=2048)
    @step
    def end(self):
        print("Scoring flow complete.")


if __name__ == '__main__':
    ScoringFlow()