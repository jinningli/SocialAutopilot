import os
import pandas as pd
import numpy as np


class DataManager():
    def __init__(self):
        self.name = "DataManager"
        self.data = None
        self.mappings = None

    def insert_dict(self, line: dict):
        raise NotImplementedError()

    def insert_dataframe(self, line: pd.DataFrame):
        raise NotImplementedError()

    def dataframe(self):
        raise NotImplementedError()

    def __str__(self):
        return "Empty Data Manager"

    def save_debug(self):
        raise NotImplementedError()


class PandasDataManager(DataManager):
    def __init__(self):
        super(DataManager, self).__init__()
        # constant
        self.name = "PandasDataManager"
        self.dtypes = {
            "user": str,
            "user.id": str,
            "user.feature": object,
            "user.label": str,  # optional
            "item": str,
            "item.id": str,
            "item.feature": object,
            "item.label": str,  # optional
            "time": float,  # Maybe use numpy.datetime64 in future https://stackoverflow.com/questions/29245848/what-are-all-the-dtypes-that-pandas-recognizes
            "time.split": int,
            "edge": str,
            "edge.feature": object,
            "edge.label": str, # optional
            "user.merge.id": int,
            "item.merge.id": int,
            "edge.merge.id": int
        }
        self.fields = list(self.dtypes.keys())
        # data storage
        self.data = pd.DataFrame(columns=self.fields)
        self.data = self.data.astype(dtype=self.dtypes)
        self.mappings = {}

    def insert_dict(self, line: dict):
        for k in line.keys():
            assert k in self.fields
        df = pd.DataFrame(line, columns=self.fields, index=[0])
        self.insert_dataframe(df)

    def insert_dataframe(self, line: pd.DataFrame):
        self.data = pd.concat([self.data, line], axis=0)

    def dataframe(self):
        return self.data

    def __str__(self):
        return str(self.data)

    def __getitem__(self, key):
        return self.data[key]

    def save_debug(self):
        self.data.to_csv("debug.csv")


if __name__ == "__main__":
    DM = DataManager()
    PDM = PandasDataManager()
    PDM.insert_dict({"user": "user001", "user.id": "iduser001", "item": "item001", "item.id": "iditem001", "time": 0.1})
    PDM.insert_dict({"user": "user002", "user.id": "iduser002", "item": "item002", "item.id": "iditem002", "time": 0.2})
    PDM.insert_dict({"user": "user002", "user.id": "iduser002",
                     "user.feature": [[0, 1, 1, 2.2, 3.3]],
                     "item": "item002", "item.id": "iditem002", "time": 0.2})
    PDM.insert_dict({"user": "user002", "user.id": "iduser002",
                     "user.feature": [[1, 2]],
                     "item": "item002", "item.id": "iditem002", "time": 0.2})
    print(PDM.data)

