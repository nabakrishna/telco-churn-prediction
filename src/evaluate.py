import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, roc_auc_score,
    ConfusionMatrixDisplay
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess_pipeline

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "evaluation")

NAVY    = "#1a1a2e"
RED     = "#ef4444"
GREEN   = "#22c55e"
BLUE    = "#3b82f6"
AMBER   = "#f59e0b"
PURPLE  = "#a78bfa"
GRAY    = "#e5e7eb"
BG      = "#f8fafc"


def load_artifacts():
    def load(name):
        with open(os.path.join(MODELS_DIR, name), "rb") as f:
            return pickle.load(f)
    model        = load("best_model.pkl")
    preprocessor = load("preprocessor.pkl")
    meta         = load("meta.pkl")
    return model, preprocessor, meta


def get_test_data(preprocessor):
    X, y, _, _, _, _, _, _ = preprocess_pipeline(DATA_PATH)
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_test, y_test


def set_style(ax, title):
    ax.set_facecolor(BG)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRAY)
    ax.tick_params(colors="#6b7280", labelsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", color=NAVY, pad=10)


def plot_roc_curve(ax, y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc     = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=BLUE, lw=2.5, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color=GRAY, lw=1.5, linestyle="--", label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.08, color=BLUE)
    ax.set_xlabel("False Positive Rate", fontsize=9, color="#6b7280")
    ax.set_ylabel("True Positive Rate", fontsize=9, color="#6b7280")
    ax.legend(fontsize=9, framealpha=0.5)
    set_style(ax, "ROC Curve")
    return roc_auc


def plot_pr_curve(ax, y_test, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)
    baseline      = y_test.mean()
    ax.plot(recall, precision, color=PURPLE, lw=2.5, label=f"AP = {avg_precision:.4f}")
    ax.axhline(baseline, color=GRAY, lw=1.5, linestyle="--", label=f"Baseline = {baseline:.2f}")
    ax.fill_between(recall, precision, alpha=0.08, color=PURPLE)
    ax.set_xlabel("Recall", fontsize=9, color="#6b7280")
    ax.set_ylabel("Precision", fontsize=9, color="#6b7280")
    ax.legend(fontsize=9, framealpha=0.5)
    set_style(ax, "Precision-Recall Curve")
    return avg_precision


def plot_f1_threshold(ax, y_test, y_prob, optimal_threshold):
    thresholds  = np.arange(0.01, 0.99, 0.01)
    f1_scores   = [f1_score(y_test, (y_prob >= t).astype(int), zero_division=0) for t in thresholds]
    prec_scores = [precision_score(y_test, (y_prob >= t).astype(int), zero_division=0) for t in thresholds]
    rec_scores  = [recall_score(y_test, (y_prob >= t).astype(int), zero_division=0) for t in thresholds]

    ax.plot(thresholds, f1_scores,   color=GREEN,  lw=2.5, label="F1 Score")
    ax.plot(thresholds, prec_scores, color=AMBER,  lw=2,   linestyle="--", label="Precision")
    ax.plot(thresholds, rec_scores,  color=RED,    lw=2,   linestyle="--", label="Recall")
    ax.axvline(optimal_threshold, color=NAVY, lw=1.5, linestyle=":", label=f"Optimal = {optimal_threshold:.2f}")

    best_f1_idx = np.argmax(f1_scores)
    ax.scatter(thresholds[best_f1_idx], f1_scores[best_f1_idx], color=GREEN, s=80, zorder=5)
    ax.annotate(f"  Best F1 = {f1_scores[best_f1_idx]:.3f}", xy=(thresholds[best_f1_idx], f1_scores[best_f1_idx]),
                fontsize=8, color=GREEN)

    ax.set_xlabel("Threshold", fontsize=9, color="#6b7280")
    ax.set_ylabel("Score", fontsize=9, color="#6b7280")
    ax.legend(fontsize=9, framealpha=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    set_style(ax, "F1 / Precision / Recall vs Threshold")


def plot_confusion_matrix(ax, y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.04)

    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=14,
                    fontweight="bold", color="white" if cm[i, j] > thresh else NAVY)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Churn", "Churn"], fontsize=9)
    ax.set_yticklabels(["No Churn", "Churn"], fontsize=9)
    ax.set_xlabel("Predicted", fontsize=9, color="#6b7280")
    ax.set_ylabel("Actual", fontsize=9, color="#6b7280")

    ax.text(0, 0, f"\nTN", ha="center", va="bottom", fontsize=7, color="#9ca3af")
    ax.text(1, 0, f"\nFP", ha="center", va="bottom", fontsize=7, color="#9ca3af")
    ax.text(0, 1, f"\nFN", ha="center", va="bottom", fontsize=7, color="#9ca3af")
    ax.text(1, 1, f"\nTP", ha="center", va="bottom", fontsize=7, color="#9ca3af")
    set_style(ax, "Confusion Matrix")


def plot_class_distribution(ax, y_test, y_pred):
    categories  = ["No Churn (Actual)", "Churn (Actual)", "No Churn (Predicted)", "Churn (Predicted)"]
    counts      = [(y_test == 0).sum(), (y_test == 1).sum(), (y_pred == 0).sum(), (y_pred == 1).sum()]
    colors      = [GREEN, RED, "#86efac", "#fca5a5"]
    bars        = ax.bar(categories, counts, color=colors, width=0.55, edgecolor="white", linewidth=1.5)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                str(count), ha="center", va="bottom", fontsize=9, fontweight="bold", color=NAVY)

    ax.set_ylabel("Count", fontsize=9, color="#6b7280")
    ax.tick_params(axis="x", labelsize=8)
    set_style(ax, "Actual vs Predicted Class Distribution")


def plot_metric_summary(ax, metrics_dict):
    ax.axis("off")
    ax.set_facecolor(BG)
    ax.set_title("Metric Summary", fontsize=11, fontweight="bold", color=NAVY, pad=10)

    labels = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    colors = [GREEN if v >= 0.75 else AMBER if v >= 0.60 else RED for v in values]

    y_positions = np.arange(len(labels))
    bar_ax = ax.inset_axes([0.05, 0.05, 0.9, 0.9])
    bars = bar_ax.barh(y_positions, values, color=colors, height=0.55, edgecolor="white")

    for bar, val in zip(bars, values):
        bar_ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=9, fontweight="bold", color=NAVY)

    bar_ax.set_yticks(y_positions)
    bar_ax.set_yticklabels(labels, fontsize=9)
    bar_ax.set_xlim(0, 1.12)
    bar_ax.set_facecolor(BG)
    bar_ax.spines[["top", "right"]].set_visible(False)
    bar_ax.spines[["left", "bottom"]].set_color(GRAY)
    bar_ax.tick_params(colors="#6b7280", labelsize=9)
    bar_ax.axvline(0.5, color=GRAY, lw=1, linestyle="--")


def plot_probability_distribution(ax, y_test, y_prob):
    churn_probs    = y_prob[y_test == 1]
    no_churn_probs = y_prob[y_test == 0]

    bins = np.linspace(0, 1, 40)
    ax.hist(no_churn_probs, bins=bins, alpha=0.6, color=GREEN, label="No Churn (Actual)", edgecolor="white")
    ax.hist(churn_probs,    bins=bins, alpha=0.6, color=RED,   label="Churn (Actual)",    edgecolor="white")
    ax.set_xlabel("Predicted Churn Probability", fontsize=9, color="#6b7280")
    ax.set_ylabel("Count", fontsize=9, color="#6b7280")
    ax.legend(fontsize=9, framealpha=0.5)
    set_style(ax, "Predicted Probability Distribution")


def run_evaluation():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading artifacts...")
    model, preprocessor, meta = load_artifacts()
    optimal_threshold = meta["threshold"]
    best_model_name   = meta["best_model_name"]

    print("Preparing test data...")
    X_test, y_test = get_test_data(preprocessor)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= optimal_threshold).astype(int)

    print("\n" + "=" * 55)
    print(f"  MODEL: {best_model_name}")
    print(f"  THRESHOLD: {optimal_threshold:.2f}")
    print("=" * 55)
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    roc_auc       = roc_auc_score(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)
    f1            = f1_score(y_test, y_pred)
    precision     = precision_score(y_test, y_pred)
    recall        = recall_score(y_test, y_pred)
    accuracy      = (y_pred == y_test).mean()

    metrics = {
        "ROC-AUC"         : roc_auc,
        "Avg Precision"   : avg_precision,
        "F1 Score"        : f1,
        "Precision"       : precision,
        "Recall"          : recall,
        "Accuracy"        : accuracy,
    }

    print("Generating evaluation plots...")

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"ChurnIQ — Model Evaluation Report\n{best_model_name}  |  Threshold: {optimal_threshold:.2f}  |  ROC-AUC: {roc_auc:.4f}",
        fontsize=14, fontweight="bold", color=NAVY, y=0.98
    )

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    ax7 = fig.add_subplot(gs[2, :])

    plot_roc_curve(ax1, y_test, y_prob)
    plot_pr_curve(ax2, y_test, y_prob)
    plot_f1_threshold(ax3, y_test, y_prob, optimal_threshold)
    plot_confusion_matrix(ax4, y_test, y_pred)
    plot_class_distribution(ax5, y_test, y_pred)
    plot_metric_summary(ax6, metrics)
    plot_probability_distribution(ax7, y_test, y_prob)

    output_path = os.path.join(OUTPUT_DIR, "evaluation_report.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()

    print(f"\nEvaluation report saved to: {output_path}")
    print("\nMetric Summary:")
    for name, val in metrics.items():
        status = "✅" if val >= 0.75 else "⚠️" if val >= 0.60 else "❌"
        print(f"  {status}  {name:<20} {val:.4f}")


if __name__ == "__main__":
    run_evaluation()