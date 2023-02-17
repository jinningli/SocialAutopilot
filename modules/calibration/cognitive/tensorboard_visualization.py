import pandas as pd
import numpy as np
import ast
from matplotlib import pyplot as plt
import random
import pylab
random.seed(0)

def tensorflow_embedding_projector(embedding, metadata: list, metadata_header=None,
                                   log_dir="./visualization/tboard_data", tag="default"):
    """
    :param embedding: numpy.array or torch tensor
    :param label: list
    :return:
    """
    import tensorflow as tf
    import tensorboard as tb
    tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
    from torch.utils.tensorboard import SummaryWriter

    print(embedding.shape[0], len(metadata))
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_embedding(embedding, metadata=metadata, metadata_header=metadata_header, tag=tag)
    writer.close()

def plot_2d_item(embedding, title=""):
    assert embedding.shape[1] == 2

    fig = plt.figure(figsize=(8, 6))

    # plot user
    emb = embedding
    emb_1 = emb[emb[:, 0] >= emb[:, 1]]
    emb_2 = emb[emb[:, 0] < emb[:, 1]]
    plt.scatter(emb_1[:, 0].reshape(-1), emb_1[:, 1].reshape(-1), marker="o", color="#178bff", label="Actor View1", s=10)
    plt.scatter(emb_2[:, 0].reshape(-1), emb_2[:, 1].reshape(-1), marker="o", color="#ff5c1c", label="Actor View2", s=10)

    plt.tick_params(labelsize=12)
    plt.legend(loc='best', prop={'size': 10})
    plt.xlabel("Belief in View1")
    plt.ylabel("Belief in View2")
    # plt.xlim([-1, 5])
    # plt.ylim([-1, 5])
    plt.title(title)
    plt.savefig("output/{}/{}_2d.jpg".format(title, title), dpi=500)

def temporal_plot(path, num=30, title=""):
    data = pd.read_csv(path)
    select_items = list(data["item"].unique())
    random.shuffle(select_items)
    select_items = select_items[:num]
    for k, item in enumerate(select_items):
        plt.cla()
        item_data = data[data["item"] == item]
        item_data = item_data.sort_values(["time"])
        previous = None
        for i, row in item_data.iterrows():
            embedding = np.array(ast.literal_eval(row["embedding"]))
            if previous is not None:
                pylab.arrow(x=previous[0], y=previous[1], dx=(embedding[0] - previous[0]), dy=(embedding[1] - previous[1]), width=0.01,
                      length_includes_head=True, facecolor="#30a5ff", edgecolor='none')
            previous = embedding
            plt.scatter(embedding[0], embedding[1], marker="o", label=row["time"], s=10)
        plt.tick_params(labelsize=12)
        plt.legend(loc='best', prop={'size': 10})
        plt.xlabel("Belief in View1")
        plt.ylabel("Belief in View2")
        plt.title(title)
        plt.savefig("output/{}/{}_{}_2d.jpg".format(title, title, k), dpi=500)


def read_embedding(path):
    data = pd.read_csv(path)
    data = data.drop_duplicates(['item'], keep='last')  # keep last
    embedding = []
    for i, row in data.iterrows():
        embedding.append(np.array(ast.literal_eval(row["embedding"])).reshape(1, -1))
    embedding = np.concatenate(embedding, axis=0)
    metadata = list(data["item"])
    return embedding, metadata

if __name__ == "__main__":
    embedding, metadata = read_embedding("./output/tgn-attn_russophobia/item_inference_result.csv")
    tensorflow_embedding_projector(embedding, metadata, log_dir="./visualization/tgn-attn_russophobia", tag="tgn-attn_russophobia")
    plot_2d_item(embedding, "tgn-attn_russophobia")
    temporal_plot("./output/tgn-attn_russophobia/item_inference_result.csv", num=30, title="tgn-attn_russophobia")

    embedding, metadata = read_embedding("./output/tgn-attn-pos_russophobia/item_inference_result.csv")
    tensorflow_embedding_projector(embedding, metadata, log_dir="./visualization/tgn-attn-pos_russophobia", tag="tgn-attn-pos_russophobia")
    plot_2d_item(embedding, "tgn-attn-pos_russophobia")
    temporal_plot("./output/tgn-attn-pos_russophobia/item_inference_result.csv", num=30, title="tgn-attn-pos_russophobia")

    embedding, metadata = read_embedding("./output/tgn-attn-pos-inner_russophobia/item_inference_result.csv")
    tensorflow_embedding_projector(embedding, metadata, log_dir="./visualization/tgn-attn-pos-inner_russophobia", tag="tgn-attn-pos-inner_russophobia")
    plot_2d_item(embedding, "tgn-attn-pos-inner_russophobia")
    temporal_plot("./output/tgn-attn-pos-inner_russophobia/item_inference_result.csv", num=30,
                  title="tgn-attn-pos-inner_russophobia")