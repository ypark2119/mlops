import pandas as pd

def preprocess():
    data = pd.read_csv("data/EcoPreprocessed.csv")
    data["tokens"] = data["review"].apply(lambda x: x.split())
    data.to_csv("data/processed_dataset.csv", index=False)

if __name__ == "__main__":
    preprocess()
