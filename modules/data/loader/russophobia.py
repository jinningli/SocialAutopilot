import os
import json
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.5f' % x)
import glob
import sys
sys.path.append("modules/data")
from DataManagement import PandasDataManager
from datetime import datetime
import time

def produce_csv(tweets_jsonl_paths):
    data = []
    for jsonl_file in tweets_jsonl_paths:
        with open(jsonl_file, "r", encoding='utf-8') as fin:
            for line in fin:
                js = json.loads(line.strip())
                t = time.mktime(datetime.strptime(js["created_at"] ,"%Y-%m-%dT%H:%M:%S.%fZ").timetuple())
                data.append([
                    js["author_id"],
                    js["author_id"],
                    js["text"].encode('utf-16', 'surrogatepass').decode('utf-16').replace(
                "\n", " ").replace("\t", " ").replace("\r", " ").replace("\"", "").replace(" ", " "),
                    js["id"],
                    time.mktime(datetime.strptime(js["created_at"] ,"%Y-%m-%dT%H:%M:%S.%fZ").timetuple())
                ])
    data = pd.DataFrame(data, columns=["user", "user.id", "item", "item.id", "time"])
    # data.sort_values(by="rawTweet")
    return data

def attach(data_manager: PandasDataManager) -> PandasDataManager:
    DATA_PATH = "data/russophobia"
    jsonl_files = []
    file_names = [x for x in glob.glob(DATA_PATH + "/*.jsonl") if os.path.basename(x).find("_") == -1]
    for file_name in sorted(file_names):
        file_name = os.path.basename(file_name)
        # for month in months:
        #     if file_name.find("{}{}".format(year, str(month).zfill(2))) != -1:
        jsonl_files.append(DATA_PATH + "/" + file_name)
    data_file = produce_csv(jsonl_files)
    data_manager.insert_dataframe(data_file)
    return data_manager


if __name__ == "__main__":
    PDM = PandasDataManager()
    PDM = attach(PDM)
    print(PDM.data)
