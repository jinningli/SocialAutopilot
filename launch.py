import yaml
import argparse
import os
import json
# import sys
# sys.path.append("modules/data")
# sys.path.append("modules/data/loader")
from modules.data.DataManagement import PandasDataManager
from modules.data.NodeMerge import NodeMergeByTokenize
from modules.data.TimeSplit import TimeSplit

parser = argparse.ArgumentParser()
# Spec config file
parser.add_argument('--config', type=str, default=None, help="Use existing yaml config to launch experiment quickly")
# Arguments
parser.add_argument('--module', type=str, default=None, help="Which module to run")
parser.add_argument('--model', type=str, default=None, help="Which model to run")
parser.add_argument('--dataset', type=str, default=None, help="Which dataset to run")
args = parser.parse_args()

def main():
    # Read config file
    if args.config is not None:
        with open(args.config, "r") as fin:
            yaml_file = yaml.safe_load(fin)
        for arg in yaml_file:
            setattr(args, arg, yaml_file[arg])
    assert args.module is not None

    # Launch Task
    # 1. Set up dataset
    if args.dataset == "russophobia":
        from modules.data.loader.russophobia import attach
    else:
        raise NotImplementedError()

    if args.module == "calibration":
        if args.model == "cognitive":
            from modules.data.saver.cognitive import save
            PDM = PandasDataManager()
            PDM = attach(PDM)
            print(PDM)
            node_merge = NodeMergeByTokenize()
            node_merge.merge(PDM, ithreshold=2, uthreshold=2)
            save(PDM, args.dataset)
        elif args.model == "dynamic_belief":
            from modules.data.saver.dynamic_belief import save
            PDM = PandasDataManager()
            PDM = attach(PDM)
            print(PDM)
            node_merge = NodeMergeByTokenize()
            node_merge.merge(PDM, ithreshold=2, uthreshold=2)
            time_split = TimeSplit()
            PDM = time_split.split(PDM, cnt=8)
            save(PDM, args.dataset)
    else:
        raise NotImplementedError()

if __name__ == "__main__":
    main()