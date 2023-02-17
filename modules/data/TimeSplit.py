import os
import math
from modules.data.DataManagement import DataManager
from modules.data.NodeMerge import NodeMergeByTokenize


class TimeSplit:
    def __init__(self):
        self.name = "TimeSplit"

    def split(self, dm: DataManager, cnt) -> DataManager:
        min_time = dm.data["time"].min()
        max_time = dm.data["time"].max()
        dm.data = dm.data.sort_values("time")
        split_length = (max_time - min_time) / cnt + 1e-5
        dm.data["time.split"] = (dm.data["time"] - min_time) // split_length
        return dm

if __name__ == "__main__":
    from DataManagement import PandasDataManager
    from loader.russophobia import attach
    PDM = PandasDataManager()
    PDM = attach(PDM)
    print(PDM.data)
    node_merge = NodeMergeByTokenize()
    PDM = node_merge.merge(PDM, ithreshold=2, uthreshold=2)
    print(PDM.data)
    time_split = TimeSplit()
    PDM = time_split.split(PDM, cnt=8)
    PDM.save_debug()
    print(PDM.data[["time", "time.split"]])