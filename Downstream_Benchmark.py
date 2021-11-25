#Copyright 2020 Vraj Shah, Arun Kumar
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from Load_Predictions import *
from downstream_models import *
from Featurize import *
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
# from autogluon import TabularPrediction as task
import copy
import warnings
warnings.filterwarnings("ignore")

# Numeric': 0,
# Categorical': 1,
# Datetime':2,
# Sentence':3,
# URL': 4,
# Numbers': 5,
# List': 6,
# Unusable': 7,
# Custom Object': 8

# classification = 1
# regression = 0
# InputFilePath = 'datasets/supreme.csv'
# TargetColumn = 'binaryClass'
# TruthVector = [1,1,1,1,1,1,1]

def downstream_benchmark(path, targetColumn, truthVector, task_type):
    file = path
    target_col = targetColumn

    if path == "datasets/flare.csv":
        dataDownstream = pd.read_csv(file, delimiter=" ")
    elif path == "datasets/articles.csv":
        dataDownstream = pd.read_csv(file, encoding="ISO-8859-1")
    else:
        dataDownstream = pd.read_csv(file)
    # print(dataDownstream.head(5))
    y = dataDownstream[[target_col]]
    dataDownstream = dataDownstream.drop(target_col, axis=1)

    dataFeaturized = FeaturizeFile(dataDownstream)

    dataFeaturized1 = ProcessStats(dataFeaturized)
    dataFeaturized2 = FeatureExtraction(dataFeaturized,dataFeaturized1,0)
    dataFeaturized2 = dataFeaturized2.fillna(0)
    # print(dataFeaturized2)

    y_RF = Load_RF(dataFeaturized2)
    y_pandas = Load_Pandas(dataDownstream)
    y_TFDV = Load_TFDV(dataDownstream)
    y_gl = Load_GLUON(dataDownstream , dataFeaturized)
    y_truth = truthVector

    attribute_names = dataDownstream.columns.values.tolist()
    # print(len(attribute_names))
    # print(attribute_names)
    # print(y_truth)
    # print(y_RF)
    # print(y_pandas)
    # print(y_TFDV)
    # print(y_gl)

    y_cur_lst = [y_truth,y_RF,y_pandas,y_TFDV,y_gl]
    model_map = {0: "truth", 1: "rf", 2: "pandas", 3: "tfdv", 4:"autogluon"}
    [print("%s:"  % model_map[i], y_cur_lst[i]) for i in range(len(y_cur_lst))]
    results = []
    for i, y_cur in enumerate(y_cur_lst):
        if i < 4:
            continue
        all_cols = Featurize(dataDownstream,attribute_names,y_cur).fillna(0)
        print("%s:"  % model_map[i], len(y_cur_lst[i]))
        print(all_cols)
        # print(all_cols.describe())
       
        # print(all_cols)

        if task_type == "classification":
            avgsc_train_lst_LR,avgsc_lst_LR,avgsc_hld_lst_LR = LogRegClassifier(all_cols,y)
            avgsc_train_lst_RF,avgsc_lst_RF,avgsc_hld_lst_RF = RandForestClassifier(all_cols,y)

        if task_type == "regression":
            avgsc_train_lst_LR,avgsc_lst_LR,avgsc_hld_lst_LR = LinearRegression(all_cols,y)
            avgsc_train_lst_RF,avgsc_lst_RF,avgsc_hld_lst_RF = RandForestRegressor(all_cols,y)
            
        print('Linear Model:')
        print(avgsc_train_lst_LR)
        print(avgsc_lst_LR)
        print(avgsc_hld_lst_LR)

        print(np.mean(avgsc_train_lst_LR))
        print(np.mean(avgsc_lst_LR))
        print(np.mean(avgsc_hld_lst_LR))

        print('Random Forest:')
        print(avgsc_train_lst_RF)
        print(avgsc_lst_RF)
        print(avgsc_hld_lst_RF)

        print(np.mean(avgsc_train_lst_RF))
        print(np.mean(avgsc_lst_RF))
        print(np.mean(avgsc_hld_lst_RF))
        result = {"model": model_map[i], "linear_train": np.mean(avgsc_train_lst_LR), "linear_test": np.mean(avgsc_lst_LR),
         "linear_holdout": np.mean(avgsc_hld_lst_LR), "rf_train": np.mean(avgsc_train_lst_RF),
         "rf_test": np.mean(avgsc_lst_RF), "rf_holdout": np.mean(avgsc_hld_lst_RF)}
        results.append(result)
    return results