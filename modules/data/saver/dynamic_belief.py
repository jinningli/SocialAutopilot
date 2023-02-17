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
    NUM_SPLIT = 8  # TODO make this argument
    WINDOW_SIZE = 3
    for t in range(NUM_SPLIT):
        if t + WINDOW_SIZE - 1 >= NUM_SPLIT:
            break
        condition = data_manager["time.split"] == t
        for k in range(1, WINDOW_SIZE):
            condition = condition | (data_manager["time.split"] == t + k)
        sub_data = data_manager[condition]
        os.makedirs("modules/calibration/dynamic_belief/data/{}".format(dataset_name), exist_ok=True)
        sub_data["label"] = [1 for _ in range(len(sub_data))]
        sub_data = sub_data.rename(columns={"user": "name", "item": "rawTweet", "item.merge.id": "tweet_id"})
        sub_data = sub_data[["label", "name", "rawTweet", "tweet_id", "time.split"]]
        sub_data.to_csv("modules/calibration/dynamic_belief/data/{}/{}_{}.csv".format(dataset_name, t, t + WINDOW_SIZE - 1), index=False, sep="\t")
    with open("modules/calibration/dynamic_belief/data/{}_mappings.json".format(dataset_name), "w") as fout:
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