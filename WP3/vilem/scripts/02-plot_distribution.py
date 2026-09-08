# %%

import matplotlib.pyplot as plt
import json
import collections
import os
import numpy as np
import seaborn as sns
import sklearn.linear_model

os.chdir(os.path.dirname(os.path.abspath(__file__))+"/..")

with open("data/judge_scores_dataset.jsonl", "r") as f:
    data = [json.loads(line) for line in f]
with open("data/pointwise_dataset.jsonl", "r") as f:
    data_human_raw = [json.loads(line) for line in f]

data_agg = collections.defaultdict(list)
for line in data:
    data_agg[line["judge"]].append(min(10, max(0, line["score"])))

data_human = []
for line in data_human_raw:
    data_human.append(line["score"]*10)

plt.rcParams["font.family"] = "serif"

fig, axs = plt.subplots(figsize=(4, 3.5), nrows=2, ncols=2, sharex=True)
for (judge, scores), ax in zip(data_agg.items(), axs.flat):
    sns.kdeplot(
        scores,
        fill=True,
        alpha=1,
        linewidth=0,
        ax=ax
    )
    sns.kdeplot(
        data_human,
        fill=False,
        linewidth=2,
        ax=ax,
        color="black",
        zorder=10
    )

    # match mean and var
    # TODO: the np.mean(scores_new) and np.std(scores_new) should be estimated based on a subset
    scores_new = np.array(scores)
    scores_new = (scores_new - np.mean(scores_new)) / np.std(scores_new) * np.std(data_human) + np.mean(data_human)
    sns.kdeplot(
        scores_new,
        fill=True,
        linewidth=0,
        alpha=0.5,
        ax=ax,
        color="tab:green",
        zorder=-10,
    )
    ax.set_title(
        judge.replace("multimodal_CoT", "").replace("_liking_text", ""),
        fontsize=9
    )
    if ax not in axs[:, 0]:
        ax.set_ylabel("")
    ax.spines[["top", "right"]].set_visible(False)

plt.xlim(0, 10)
plt.tight_layout(pad=0.1)
plt.show()