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
    if line["judge"] not in [
        # "qwen25omni_success_text",
        "phi4multimodal_CoT_summary_text",
        "phi4multimodal_CoT_questions_summary_liking_text",
        "phi4multimodal_CoT_questions_speech",
    ]:
        continue
    data_agg[line["judge"]].append(min(10, max(0, line["score"])))

data_human = []
for line in data_human_raw:
    data_human.append(line["score"]*10)

plt.rcParams["font.family"] = "serif"

fig, axs = plt.subplots(figsize=(4.2, 1.5), nrows=1, ncols=2, sharex=True, sharey=True)
sns.kdeplot(
    data_human,
    fill=False,
    linewidth=2,
    ax=axs[0],
    color="black",
    zorder=10
)
sns.kdeplot(
    data_human,
    fill=False,
    linewidth=2,
    ax=axs[1],
    color="black",
    zorder=10
)



for (judge, scores), color in zip(list(data_agg.items())[::-1], ["tab:blue", "tab:green", "tab:orange"]):
    sns.kdeplot(
        scores,
        fill=True,
        alpha=0.5,
        linewidth=0,
        bw_adjust=1,
        ax=axs[0],
        color=color,
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
        bw_adjust=1,
        ax=axs[1],
        color=color,
        zorder=-10,
    )
text_kwargs = dict(
    fontsize=7,
    ha='center', va='center',
)

for ax in axs:
    ax.set_yticks([])
    if ax != axs[0]:
        ax.set_ylabel("")
    ax.spines[["top", "right"]].set_visible(False)

    ax.text(
        0.15, 0.08,
        "human",
        transform=ax.transAxes,
        **text_kwargs
    )

axs[0].text(
    0.55, 0.9,
    "Qwen2.5",
    color="tab:orange",
    transform=axs[0].transAxes,
    **text_kwargs
)
axs[0].text(
    0.85, 0.5,
    "Phi4 A",
    color="tab:blue",
    transform=axs[0].transAxes,
    **text_kwargs
)
axs[0].text(
    0.95, 0.4,
    "Phi4 B",
    color="tab:green",
    transform=axs[0].transAxes,
    **text_kwargs
)

axs[1].text(
    0.85, 0.7,
    "Qwen2.5",
    color="tab:orange",
    transform=axs[1].transAxes,
    **text_kwargs
)
axs[1].text(
    0.85, 0.6,
    "Phi4 A",
    color="tab:blue",
    transform=axs[1].transAxes,
    **text_kwargs
)
axs[1].text(
    0.85, 0.5,
    "Phi4 B",
    color="tab:green",
    transform=axs[1].transAxes,
    **text_kwargs
)

axs[0].set_xlabel("Score (raw)")
axs[1].set_xlabel("Score (calibrated)")
plt.xticks([0, 2, 4, 6, 8, 10])
plt.xlim(0, 10)
plt.tight_layout(pad=0.2)
os.makedirs("computed/", exist_ok=True)
plt.savefig("computed/score_distribution.pdf")
plt.show()