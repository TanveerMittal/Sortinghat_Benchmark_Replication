import ast
import sys
import pandas as pd
from Downstream_Benchmark import downstream_benchmark

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        metadata = pd.read_csv("testdata/test.csv")
    else:
        metadata = pd.read_csv("Metadata/metadata.csv")
    results_path = "results/results.csv"
    results_df = pd.DataFrame(columns=["name", "model", "linear_train", "linear_test", "linear_holdout",
                                       "rf_train", "rf_test", "rf_holdout"])
    datasets_with_errors = [19, 22, 23, 24, 29]
    for i in range(len(metadata)):
        if i in datasets_with_errors:
            continue
        curr_data = metadata.loc[i]
        name = curr_data["name"].lower()
        print("Dataset:", name)
        file_path = "datasets/%s.csv" % curr_data["name"].lower()
        target_col = curr_data["target"]
        task = curr_data["task"]
        truth_vector = ast.literal_eval(curr_data["truthVector"])
        results = downstream_benchmark(file_path, target_col, truth_vector, task)
        for res in results:
            res["name"] = name
            results_df = results_df.append(res, ignore_index=True)
        results_df.to_csv(results_path, index=False)
    results_df.to_csv(results_path, index=False)
