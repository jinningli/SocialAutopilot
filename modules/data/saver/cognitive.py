import os
import pandas as pd
import json
import numpy as np
import sys
sys.path.append("modules/data")
from DataManagement import DataManager
from DataManagement import PandasDataManager
from NodeMerge import NodeMergeByTokenize

# Setting the random seeds
np.random.seed(0)

def save(data_manager: DataManager, dataset_name: str) -> DataManager:
    with open("modules/calibration/cognitive/data/{}.csv".format(dataset_name), "w") as fout:
        data_manager.data = data_manager.data.sort_values(["time"])  # important to sort by time
        fout.write("user_id,item_id,timestamp,state_label,comma_separated_list_of_features\n")
        num_user = len(data_manager.data["user.merge.id"])
        num_item = len(data_manager.data["item.merge.id"])
        num_edge = len(data_manager.data["edge.merge.id"])
        feature_table = np.random.random(size=(num_edge, 128))  # currently use random feature
        for i, row in data_manager.data.iterrows():
            line = [str(row["user.merge.id"]), str(row["item.merge.id"]), str(row["time"]), "0"]
            feature = feature_table[row["edge.merge.id"]].tolist()
            line.append(",".join([str(f) for f in feature]))
            fout.write(",".join(line) + "\n")
    with open("modules/calibration/cognitive/data/{}_mappings.json".format(dataset_name), "w") as fout:
        json.dump(data_manager.mappings, fout, indent=2)


if __name__ == "__main__":
    from DataManagement import PandasDataManager
    from loader.russophobia import attach
    PDM = PandasDataManager()
    PDM = attach(PDM)
    print(PDM.data)
    node_merge = NodeMergeByTokenize()
    node_merge.merge(PDM, ithreshold=2, uthreshold=2)
    print(PDM.data)
    PDM.save_debug()
    save(PDM, "russophobia")