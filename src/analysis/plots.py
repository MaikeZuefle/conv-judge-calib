import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_distributions(scores, labels, path):
    normalized_scores = [score / 10 for score in scores]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(normalized_scores, bins=11, range=(0, 1))
    axes[0].set_title("predicted scores")
    axes[0].set_xlabel("judge score (normalized)")
    axes[1].hist(labels, bins=11, range=(0, 1))
    axes[1].set_title("true labels")
    axes[1].set_xlabel("paper_pcs_proxy")
    fig.tight_layout()
    fig.savefig(path)
