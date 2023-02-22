import os
import random

import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
import pylab
import matplotlib
from wordcloud import WordCloud, STOPWORDS
from tqdm import tqdm
import pickle

NOISE = 0.04

def dynamic_plot(x, y, times):
    for i in range(x.shape[0] - 1):
        pylab.arrow(x=x[i], y=y[i], dx=(x[i+1] - x[i]), dy=(y[i+1] - y[i]), width=0.005,
                    # width=min(np.average(x), np.average(y)) * 0.01,
                  length_includes_head=True, facecolor="#30a5ff", edgecolor='none')
    for i in range(x.shape[0]):
        # [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
        color_map = {
            "0_2": "#1f77b4",
            "1_3": "#ff7f0e",
            "2_4": "#2ca02c",
            "3_5": "#d62728",
            "4_6": "#9467bd",
            "5_7": "#8c564b"
        }
        plt.scatter(x[i], y[i], marker="o", label=times[i], s=10, c=color_map[times[i]])


def dynamic_plot_all_in_one(x, y, user_id, color):
    for i in range(x.shape[0] - 1):
        pylab.arrow(x=x[i], y=y[i], dx=(x[i+1] - x[i]), dy=(y[i+1] - y[i]), width=0.005,
                    # width=min(np.average(x), np.average(y)) * 0.01,
                  length_includes_head=True, facecolor=color, edgecolor='none')
    for i in range(x.shape[0]):
        if i == 0:
            plt.scatter(x[i], y[i], marker="o", s=10, c=color, label=user_id)
        else:
            plt.scatter(x[i], y[i], marker="o", s=10, c=color)


def run(times, n=20, selected_users=None):
    output_dir = "../output/russophobia"
    os.makedirs(output_dir + "/trajectory", exist_ok=True)

    userlists = []
    asserlists = []
    embeddinglists = []
    for time in times:
        with open(output_dir + "/" + time + "/" + "namelist.json", "r") as fin:
            userlist = json.load(fin)
            userlists.append(userlist)
        with open(output_dir + "/" + time + "/" + "asserlist.json", "r") as fin:
            asserlist = json.load(fin)
            asserlists.append(asserlist)
        with open(output_dir + "/" + time + "/" + "embedding.bin", "rb") as fin:
            embedding = pickle.load(fin)
            embeddinglists.append(embedding)

    if selected_users is not None:
        overlapping_users = selected_users
    else:
        overlapping_users = set(userlists[0])
        for userlist in userlists[1:]:
            overlapping_users = overlapping_users.intersection(set(userlist))

        overlapping_users = list(overlapping_users)
        random.shuffle(overlapping_users)

    for user in overlapping_users[:n]:
        plt.cla()
        fig = plt.figure(figsize=(8, 6))
        x = []
        y = []
        for i, time in enumerate(times):
            embedding = embeddinglists[i][userlists[i].index(user)]
            x.append(embedding[0])
            y.append(embedding[1])
        x = np.array(x)
        y = np.array(y)
        dynamic_plot(x, y, times)
        plt.tick_params(labelsize=12)
        plt.legend(loc='best', prop={'size': 10})
        plt.xlabel("Belief in View1")
        plt.ylabel("Belief in View2")
        plt.title("User ID: {}".format(user))
        plt.xlim([-0.1, 2])
        plt.ylim([-0.1, 2])
        plt.savefig(output_dir + "/trajectory/{}.jpg".format(user), dpi=500)

        plt.cla()
        if np.sum(x >= y) > np.sum(x < y):
            sequence = x
            plt.plot(times, sequence, label="Influencer Belief", marker='^')
            plt.ylabel("Belief in View 1")
            plt.xlabel("Time (Month)")
            plt.legend()
        else:
            sequence = y
            plt.plot(times, sequence, label="Influencer Belief", marker='^')
            plt.ylabel("Belief in View 2")
            plt.xlabel("Time (Month)")
            plt.legend()
        plt.title("ID: {}".format(user))
        plt.xlim([-0.1, 2])
        plt.ylim([-0.1, 2])
        plt.savefig(output_dir + "/trajectory/{}_curve.jpg".format(user), dpi=500)

        his_tweets = []
        for time in times:
            data_df = pd.read_csv("../data/russophobia/{}.csv".format(time),
                                  sep='\t', low_memory=False, dtype={"name": str})
            data_df = data_df[data_df["name"] == user]
            data_df["time"] = [time for _ in range(len(data_df))]
            his_tweets.append(data_df)
        his_tweets = pd.concat(his_tweets, axis=0)
        his_tweets.to_csv(output_dir + "/trajectory/{}.csv".format(user), index=False, sep="\t")

        if len(his_tweets) == 0:
            continue

        # plot wordcloud
        comment_words = u''
        stopwords = set(STOPWORDS)
        for val in his_tweets.rawTweet:
            original_text = str(val).replace("RT @", "").lower()
            tok = original_text.split(' ')
            for x in tok:
                if len(x) == 0:
                    continue
                elif x[0:4] == 'http' or x[0:5] == 'https':
                    continue
                elif x[0] == '@':
                    continue
                elif x[0] == '#':
                    continue
                comment_words = comment_words + ' ' + x

        wordcloud = WordCloud(width=2560, height=1920,
                              background_color='white',
                              stopwords=stopwords,
                              min_font_size=10, colormap=matplotlib.cm.get_cmap('GnBu')).generate(comment_words)
        plt.figure(figsize=(9, 6), facecolor=None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(output_dir + "/trajectory/{}_cloud.jpg".format(user), dpi=500)


def arrow_plot_all_in_one(times, n=20):
    output_dir = "../output/russophobia"
    os.makedirs(output_dir + "/trajectory", exist_ok=True)

    userlists = []
    asserlists = []
    embeddinglists = []
    for time in times:
        with open(output_dir + "/" + time + "/" + "namelist.json", "r") as fin:
            userlist = json.load(fin)
            userlists.append(userlist)
        with open(output_dir + "/" + time + "/" + "asserlist.json", "r") as fin:
            asserlist = json.load(fin)
            asserlists.append(asserlist)
        with open(output_dir + "/" + time + "/" + "embedding.bin", "rb") as fin:
            embedding = pickle.load(fin)
            embeddinglists.append(embedding)

    overlapping_users = set(userlists[0])
    for userlist in userlists[1:]:
        overlapping_users = overlapping_users.intersection(set(userlist))

    overlapping_users = list(overlapping_users)
    random.shuffle(overlapping_users)
    fig = plt.figure(figsize=(8, 6))
    colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
    for p, user in enumerate(overlapping_users[:n]):
        x = []
        y = []
        for i, time in enumerate(times):
            embedding = embeddinglists[i][userlists[i].index(user)]
            x.append(embedding[0])
            y.append(embedding[1])
            if x[-1] > y[-1]:
                y[-1] += np.random.rand() * NOISE
            else:
                x[-1] += np.random.rand() * NOISE
        x = np.array(x)
        y = np.array(y)
        dynamic_plot_all_in_one(x, y, user, colors[p % len(colors)])
    plt.tick_params(labelsize=12)
    plt.legend(loc='best', prop={'size': 10})
    plt.xlabel("Belief in View1")
    plt.ylabel("Belief in View2")
    plt.title("Trajectory of {} users, Times: {}".format(n, times))
    plt.xlim([-0.1, 2])
    plt.ylim([-0.1, 2])
    plt.savefig(output_dir + "/all_in_one.jpg", dpi=500)

    return overlapping_users

selected_users = arrow_plot_all_in_one(times=["0_2", "5_7"], n=10)
# run(times=["0_2", "1_3", "2_4", "3_5", "4_6", "5_7"], selected_users=selected_users)



