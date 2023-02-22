import os

dataset_name = "russophobia"
USE_FREEZE = True
times = ["0_2", "1_3", "2_4", "3_5", "4_6", "5_7"]
for i, time in enumerate(times):
    print("Remember to delete all pkl in ./data")
    if USE_FREEZE:
        command = "python3 main.py --data_path data/{}/{}.csv --dataset {} " \
                  "--stopword_path stopwords_en.txt --model InfoVGAE --hidden2_dim 2 --learning_rate 0.1 " \
                  "--seed 0 --epoch 200 --kthreshold 2 --uthreshold 2 --output_path ./output/{}/{} ".format(
            dataset_name, time, dataset_name, dataset_name, time
        )
        if i != 0:
            command += "--freeze_dict ./output/{}/{}/freeze_dict.pkl".format(dataset_name, times[i - 1])
        print(command)
        os.system(command)
    else:
        command = "python3 main.py --data_path data/{}/{}.csv --dataset {}_init " \
                  "--stopword_path stopwords_en.txt --model InfoVGAE --hidden2_dim 2 --learning_rate 0.1 " \
                  "--seed 0 --epoch 200 --kthreshold 2 --uthreshold 2 --output_path ./output/{}/{}".format(
            dataset_name, time, dataset_name, dataset_name, time
        )
        print(command)

