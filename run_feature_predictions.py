import ast
import sys
import pandas as pd
from Load_Predictions import *
from Featurize import *

def get_predictions(file_path, target_col, truth_vector):
    if file_path == "datasets/flare.csv":
        dataDownstream = pd.read_csv(file_path, delimiter=" ")
    elif file_path == "datasets/articles.csv":
        dataDownstream = pd.read_csv(file_path, encoding="ISO-8859-1")
    else:
        dataDownstream = pd.read_csv(file_path)
    y = dataDownstream[[target_col]]
    dataDownstream = dataDownstream.drop(target_col, axis=1)

    dataFeaturized = FeaturizeFile(dataDownstream)

    dataFeaturized1 = ProcessStats(dataFeaturized)
    dataFeaturized2 = FeatureExtraction(dataFeaturized,dataFeaturized1,0)
    dataFeaturized2 = dataFeaturized2.fillna(0)

    y_RF = Load_RF(dataFeaturized2)
    y_pandas = Load_Pandas(dataDownstream)
    y_TFDV = Load_TFDV(dataDownstream)
    y_gl = Load_GLUON(dataDownstream , dataFeaturized)
    y_truth = truth_vector

    return {"truth": y_truth, "rf": y_RF,
            "pandas":y_pandas, "tfdv":y_TFDV, "agl":y_gl}
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        metadata = pd.read_csv("testdata/test.csv")
    else:
        metadata = pd.read_csv("Metadata/metadata.csv")
    results_path = "results/predictions.csv"
    results_df = pd.DataFrame(columns=["dataset", "truth", "rf", "pandas", "tfdv", "agl"])
    for i in range(len(metadata)):
        curr_data = metadata.loc[i]
        name = curr_data["name"].lower()
        print("Dataset:", name)
        file_path = "datasets/%s.csv" % curr_data["name"].lower()
        target_col = curr_data["target"]
        task = curr_data["task"]
        truth_vector = ast.literal_eval(curr_data["truthVector"])
        results = get_predictions(file_path, target_col, truth_vector)
        results["dataset"] = name
        results_df = results_df.append(results, ignore_index=True)
        results_df.to_csv(results_path, index=False)