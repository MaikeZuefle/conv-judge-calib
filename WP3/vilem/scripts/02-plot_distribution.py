# %%

import math
from typing import Callable

import matplotlib.pyplot as plt
import json
import collections
import os
import numpy as np
import seaborn as sns
import statistics
import sklearn.linear_model
import scipy.stats

os.chdir(os.path.dirname(os.path.abspath(__file__))+"/..")

with open("data/judge_scores_dataset.jsonl", "r") as f:
    data_llm_raw = [json.loads(line) for line in f]
with open("data/pointwise_dataset.jsonl", "r") as f:
    data_human_raw = [json.loads(line) for line in f]

data_agg = collections.defaultdict(dict)
for line in data_llm_raw:
    data_agg[line["judge"]][line["conversation"]] = min(10, max(0, line["score"]))

data_human = {}
for line in data_human_raw:
    data_human[line["conversation"]] = line["score"]*10

common_conversations = list(set.intersection(set(data_human.keys()), *[set(scores.keys()) for scores in data_agg.values()]))
data_human = [data_human[conv] for conv in common_conversations]
data_agg = {
    judge: [scores[conv] for conv in common_conversations]
    for judge, scores in data_agg.items()
}

def loss_mse(y_human: list[float], y_llm: list[float]) -> float:
    return statistics.mean([abs(a - b)**2 for a, b in zip(y_human, y_llm)])

def loss_bucket_sizes(y_human: list[float], y_llm: list[float]) -> float:
    bins = np.linspace(0, 10, 11)
    hist_human, _ = np.histogram(y_human, bins=bins)
    hist_llm, _ = np.histogram(y_llm, bins=bins)
    hist_human = hist_human / sum(hist_human)
    hist_llm = hist_llm / sum(hist_llm)
    return statistics.mean([abs(a - b)**2 for a, b in zip(hist_human, hist_llm)])

def loss_wasserstein(y_human: list[float], y_llm: list[float]) -> float:
    return scipy.stats.wasserstein_distance(y_human, y_llm)

def transform_identity(y_human: list[float], y_llm: list[float], y_llm_test: list[float]) -> list[float]:
    return y_llm_test

def transform_constant(y_human: list[float], y_llm: list[float], y_llm_test: list[float]) -> list[float]:
    b = statistics.mean([a - b for a, b in zip(y_human, y_llm)])
    return [y + b for y in y_llm_test]

def transform_linear(y_human: list[float], y_llm: list[float], y_llm_test: list[float]) -> list[float]:
    model = sklearn.linear_model.LinearRegression(fit_intercept=False)
    model.fit(np.array(y_llm).reshape(-1, 1), np.array(y_human).reshape(-1, 1))
    return model.predict(np.array(y_llm_test).reshape(-1, 1)).flatten().tolist()

def transform_affine(y_human: list[float], y_llm: list[float], y_llm_test: list[float]) -> list[float]:
    model = sklearn.linear_model.LinearRegression(fit_intercept=True)
    model.fit(np.array(y_llm).reshape(-1, 1), np.array(y_human).reshape(-1, 1))
    return model.predict(np.array(y_llm_test).reshape(-1, 1)).flatten().tolist()

def transform_minmax(y_human: list[float], y_llm: list[float], y_llm_test: list[float]) -> list[float]:
    min_human, max_human = min(y_human), max(y_human)
    min_llm, max_llm = min(y_llm), max(y_llm)
    if max_llm - min_llm == 0:
        return y_llm_test
    return [
        (y - min_llm) / (max_llm - min_llm) * (max_human - min_human) + min_human
        for y in y_llm_test
    ]

def transform_musigma(y_human: list[float], y_llm: list[float], y_llm_test: list[float]) -> list[float]:
    mean_human, std_human = statistics.mean(y_human), statistics.stdev(y_human)
    mean_llm, std_llm = statistics.mean(y_llm), statistics.stdev(y_llm)
    if std_llm == 0:
        return y_llm_test
    return [
        (y - mean_llm) / std_llm * std_human + mean_human
        for y in y_llm_test
    ]

def get_anchors_set_random(y_human: list[float], size: int) -> list[int]:
    return np.random.choice(len(y_human), size=size, replace=False).tolist()


def get_anchors_set_optimized(
    y_human: list[float], y_llms: list[list[float]],
    size: int,
    fn: Callable[[list[float], list[float], list[float]], list[float]]
) -> list[int]:
    best_indices = None
    best_mae = math.inf
    for _ in range(1_000):
        indices = get_anchors_set_random(y_human, size=size)
        scores_human = [y_human[i] for i in indices]
        losses = []
        for scores in y_llms:
            scores_anchors = [scores[i] for i in indices]
            scores_new = np.clip(fn(scores_human, scores_anchors, scores), 0, 10).tolist()
            loss = loss_wasserstein(y_human, scores_new)
            losses.append(loss)
        loss = statistics.mean(losses)
        if loss < best_mae:
            best_mae = loss
            best_indices = indices

    assert best_indices is not None
    return best_indices

import scipy.optimize

def transform_bucket_optimized(y_human: list[float], y_llm: list[float], y_llm_test: list[float]) -> list[float]:
    def loss(params):
        a, b = params
        y_llm_transformed = [a * y + b for y in y_llm]
        return loss_wasserstein(y_human, y_llm_transformed)
    
    # Initialize with the constant shift as a stable prior
    b_init = statistics.mean([a - b for a, b in zip(y_human, y_llm)])
    initial_guess = [1.0, b_init] 
    
    result = scipy.optimize.minimize(loss, initial_guess, method='Nelder-Mead')
    
    a_opt, b_opt = result.x
    return [a_opt * y + b_opt for y in y_llm_test]

METHODS_TRANSFORM = [
    ("identity", transform_identity),
    ("constant", transform_constant),
    ("linear", transform_linear),
    ("affine", transform_affine),
    ("minmax", transform_minmax),
    ("musigma", transform_musigma),
    ("bucket_optimized", transform_bucket_optimized),
]


for method_transform_name, method_transform in METHODS_TRANSFORM:
    for size in [8, 32, 128]:
        maes_random = []
        maes_optimized = []

        # random selection
        for _ in range(10):
            indices = get_anchors_set_random(data_human, size=size)
            scores_human = [data_human[i] for i in indices]
            for judge, scores in data_agg.items():
                scores_anchors = [scores[i] for i in indices]
                scores_new = np.clip(method_transform(scores_human, scores_anchors, scores), 0, 10).tolist()
                maes_random.append(loss_wasserstein(data_human, scores_new))

        # optimized selection
        for judge, scores in data_agg.items():
            indices = get_anchors_set_optimized(
                data_human,
                [scores for _judge, scores in data_agg.items() if _judge != judge],
                size=size,
                fn=method_transform
            )
            scores_human = [data_human[i] for i in indices]
            scores_anchors = [scores[i] for i in indices]
            scores_new = np.clip(method_transform(scores_human, scores_anchors, scores), 0, 10).tolist()
            maes_optimized.append(loss_wasserstein(data_human, scores_new))

        print(f"{method_transform_name:<10} {size:<5} {statistics.mean(maes_random):>5.3}  {statistics.mean(maes_optimized):>5.3}")


# %%
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

data_agg_local = {
    k: v
    for k, v in data_agg.items()
    if k not in {
        # "qwen25omni_success_text",
        # "phi4multimodal_CoT_summary_text",
        # "phi4multimodal_CoT_questions_summary_liking_text",
        # "phi4multimodal_CoT_questions_speech",
    }
}

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

    # mu/sigma calibration
    scores_new = np.clip(transform_constant(data_human, scores, scores), 0, 10).tolist()

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
    0.95, 0.7,
    "Qwen2.5",
    color="tab:orange",
    transform=axs[1].transAxes,
    **text_kwargs
)
axs[1].text(
    0.92, 0.6,
    "Phi4 A",
    color="tab:blue",
    transform=axs[1].transAxes,
    **text_kwargs
)
axs[1].text(
    0.92, 0.5,
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