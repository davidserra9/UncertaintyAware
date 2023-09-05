import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.patches as mpl_patches
from matplotlib.patches import Patch
from sklearn.metrics import accuracy_score, auc

def compute_metrics(y_true, y_pred, cls_names):
    f1 = compute_f1(y_true, y_pred)
    acc = compute_accuracy(y_true, y_pred)
    cm = compute_confusion_matrix(y_true, y_pred, cls_names)

    return f1, acc, cm

def compute_f1(y_true, y_pred, average='macro') -> float:
    return f1_score(y_true, y_pred, average=average)

def compute_accuracy(y_true, y_pred) -> float:
    return np.mean(y_true == y_pred)

def compute_confusion_matrix(y_true, y_pred, cls_names) -> plt.Figure:
    conf_mat = confusion_matrix(y_true, y_pred)     # Compute CM
    conf_mat_norm = conf_mat / conf_mat.sum(axis=1, keepdims=True)   # Normalize CM

    cls_names = [c.replace('_', '\n').replace(' ', '\n').capitalize() for c in cls_names]   # Clean names

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(conf_mat_norm,
                annot=True,
                fmt='.2%',
                cmap=plt.get_cmap('Greys'),
                annot_kws={"size": 13},
                ax=ax)

    # x ticks labels
    ax.set_xticks([i + 0.5 for i in range(len(cls_names))])
    ax.set_xticklabels(cls_names, fontsize=10, fontstyle='italic')
    ax.set_xlabel('Predicted Values', size=13)

    # y ticks labels
    ax.set_yticks([i + 0.5 for i in range(len(cls_names))])
    ax.set_yticklabels(cls_names, fontsize=10, fontstyle='italic', rotation=45)
    ax.set_ylabel('GT Values', size=13)

    # title = f"Confusion Matrix"
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.tick_params(labelsize=10, length=0)

    samples = conf_mat.flatten().tolist()
    samples = [str(s) for s in samples]
    # samples = ['' for s in samples if s=='0']
    # samples = samples.replace('0', '')

    for text_elt, additional_text in zip(ax.texts, samples):
        ax.text(*text_elt.get_position(), '\n' + additional_text, color=text_elt.get_color(),
                ha='center', va='top', size=13)
    plt.close()
    return fig

def predictive_entropy(mean):
    """ Function to compute the predictive entropy of the network
        Parameters
        ----------
        mean : np.array
            mean of the MC samples with shape (I, C) or (I, N, C)
            I: total number of input annotations
            N: number of images per annotation
            C: number of classes
        Return
        ------
        predictive_entropy : np.array
            predictive entropy of the network with shape (I,)
            I: total number of input annotations
    """

    epsilon = sys.float_info.min
    if len(mean.shape) == 1:
        return -np.sum(mean * np.log(mean + epsilon))
    elif len(mean.shape) == 2:
        return -np.sum(mean * np.log(mean + epsilon), axis=-1)
    else:
        return -np.sum(np.mean(mean, axis=1) * np.log(np.mean(mean, axis=1) + epsilon), axis=-1)

def histogram_intersection(data1, data2, nbins=100):
    """ Function to compute the intersection between two data samples
        Parameters
        ----------
        data1: np.array
            data array
        data2: np.array
            data array
        nbins: int
            number of bins
        Returns
        -------
        histogram intersection: int
    """
    h1 = np.histogram(data1, density=True, bins=[i * 0.01 for i in range(0, nbins + 1)])[0]
    h2 = np.histogram(data2, density=True, bins=[i * 0.01 for i in range(0, nbins + 1)])[0]

    return np.sum(np.minimum(h1, h2)) / 100

def uncertainty_box_plot(y_true, y_pred, **metrics):
    """ Function to compute and plot the box plots and histograms from uncertainty estimations split in correct and
        incorrect samples.
    Parameters
    ----------
    y_true: list
        ground truth labels
    y_pred: list
        predicted labels
    metrics: np.array
        predicted uncertainty estimations. They can be obtained from different metrics (std, entropy, BC...)
    Returns
    -------
    matplotlib.pyplot.Figure
    """

    # correct/incorrect color palette ('green', 'red', 'green', 'red' ...)
    sns.set_palette(sns.color_palette("prism"))

    # create a dataframe which the desired data(metrics, y_true, y_pred, status - correct/incorrect)
    df_data = metrics.copy()
    for key, value in {"y_true": y_true,
                       "y_pred": y_pred,
                       "status": np.where((y_true == y_pred) == True, "correct", "incorrect")}.items():
        df_data[key] = value
    df = pd.DataFrame(df_data)

    fig, axes = plt.subplots(2, len(metrics), figsize=(5*len(metrics), 8))
    for idx, (key, value) in enumerate(metrics.items()):
        subfig = sns.boxplot(data=df,
                             y=key,
                             x="status",
                             showfliers=True,
                             width=0.35,
                             whis=[0, 100],
                             ax=axes[0] if len(metrics)==1 else axes[0, idx])
        subfig.set(xlabel=None)
        subfig.set(ylabel=None)
        subfig.set(xticklabels=[])
        subfig.set_title(key.replace("_", " ").capitalize())

    for idx, (key, value) in enumerate(metrics.items()):
        subfig = sns.histplot(np.ma.array(df[key], mask=np.invert(y_true == y_pred)).compressed(), stat="probability",
                              kde=True, color=sns.color_palette("prism")[0], line_kws={'linewidth': 3}, bins=100,
                              ax=axes[1] if len(metrics) == 1 else axes[1, idx])
        subfig = sns.histplot(np.ma.array(df[key], mask=(y_true == y_pred)).compressed(), stat="probability", kde=True,
                              color=sns.color_palette("prism")[1], line_kws={'linewidth': 3}, bins=100,
                              ax=axes[1] if len(metrics) == 1 else axes[1, idx])

        hist_intersection = histogram_intersection(np.ma.array(df[key], mask=np.invert(y_true == y_pred)).compressed(),
                                                   np.ma.array(df[key], mask=(y_true == y_pred)).compressed())

        subfig.legend([mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)],
                      ["C" + r"$\cap$" + f"I: {hist_intersection:.4}"],
                      loc="best", fontsize=12, fancybox=True, shadow=True, handlelength=0, handletextpad=0)

        subfig.set(xlabel=None)
        subfig.set(ylabel=None)

    custom_lines = [Patch(facecolor=sns.color_palette("prism")[0]),
                    Patch(facecolor=sns.color_palette("prism")[1])]

    if len(metrics) == 1:
        axes[0].legend(custom_lines, ["correct", "incorrect"], fontsize=12, fancybox=True, shadow=True)
        axes[0].set_ylabel("Uncertainty values", size="large")
        axes[1].set_ylabel("Normalized KDE/Probability histogram", size="large")
    else:
        axes[0, 0].legend(custom_lines, ["correct", "incorrect"], fontsize=12, fancybox=True, shadow=True)
        axes[0, 0].set_ylabel("Uncertainty values", size="large")
        axes[1, 0].set_ylabel("Normalized KDE/Probability histogram", size="large")
    fig.tight_layout()
    plt.close()
    return fig, hist_intersection

def correct_incorrect_histogram(y_true, y_pred, **metrics):
    """ Function to compute and plot the box plots and histograms from uncertainty estimations split in correct and
        incorrect samples.
    Parameters
    ----------
    y_true: list
        ground truth labels
    y_pred: list
        predicted labels
    metrics: np.array
        predicted uncertainty estimations. They can be obtained from different metrics (std, entropy, BC...)
    Returns
    -------
    matplotlib.pyplot.Figure
    """
    if len(metrics) > 1:
        raise ValueError("This function only supports one metric at a time")

    # correct/incorrect color palette ('green', 'red', 'green', 'red' ...)
    sns.set_palette(sns.color_palette("prism"))

    # create a dataframe which the desired data(metrics, y_true, y_pred, status - correct/incorrect)
    df_data = metrics.copy()
    for key, value in {"y_true": y_true,
                       "y_pred": y_pred,
                       "status": np.where((y_true == y_pred) == True, "correct", "incorrect")}.items():
        df_data[key] = value
    df = pd.DataFrame(df_data)

    fig, ax = plt.subplots(figsize=(6, 6))

    sns.histplot(np.ma.array(df['entropy'], mask=np.invert(y_true == y_pred)).compressed(), stat="probability",
                          kde=True, color=sns.color_palette("prism")[0], line_kws={'linewidth': 3}, bins=100, ax=ax)
    sns.histplot(np.ma.array(df['entropy'], mask=(y_true == y_pred)).compressed(), stat="probability", kde=True,
                          color=sns.color_palette("prism")[1], line_kws={'linewidth': 3}, bins=100, ax=ax)


    hist_intersection = histogram_intersection(np.ma.array(df['entropy'], mask=np.invert(y_true == y_pred)).compressed(),
                                               np.ma.array(df['entropy'], mask=(y_true == y_pred)).compressed())

    custom_lines = [Patch(facecolor=sns.color_palette("prism")[0]),
                    Patch(facecolor=sns.color_palette("prism")[1])]

    plt.legend(custom_lines, [f"Correct (avg: {df[df['status'] =='correct']['entropy'].mean():.2f})",
                              f"Incorrect (avg: {df[df['status'] =='incorrect']['entropy'].mean():.2f})"],
               fontsize=12, fancybox=True, shadow=True, loc='best')

    plt.ylabel("Normalized KDE/Probability histogram", size="large")
    plt.xlabel("Predictive Entropy", size="large")

    fig.tight_layout()
    plt.close()
    return fig, hist_intersection

def uncertainty_curve(y_true, y_pred, ascending=True, **metrics):
    """ Function to compute and plot the Uncertainty Ordering Curve and the corresponding areas under the curve.
    Parameters
    ----------
    y_true: list
        ground truth labels
    y_pred: list
        predicted labels
    metrics: np.array
        predicted uncertainty estimations. They can be obtained from different metrics (std, entropy, BC...)
    Returns
    -------
    matplotlib.pyplot.Figure
    """

    sns.set_palette(sns.color_palette("Set1"))
    fig = plt.figure(figsize=(6, 6))
    for name, (metric, y_pred) in metrics.items():
        accuracy = []

        # sort predictions by uncertainty
        metric, y_true_ord, y_pred_ord = (list(t) for t in zip(*sorted(zip(metric, y_true, y_pred), reverse=ascending)))

        for idx in range(len(y_true_ord)):
            accuracy.append(accuracy_score(y_true_ord, y_pred_ord))
            y_pred_ord[idx] = y_true_ord[idx]

        # au = auc(np.array(range(len(accuracy))) / len(accuracy), accuracy)
        # nau = au / auc(np.array(range(len(ideal_curve))) / len(ideal_curve), ideal_curve)

        plt.plot((np.array(range(len(accuracy))) * 100) / len(accuracy),
                 accuracy,
                 linewidth=2,
                 label=f"{name.replace('_', ' ').capitalize()}")

    plt.xlabel("Percentage of corrected samples (%)")
    plt.ylabel("F1-Score")
    plt.legend(fontsize=12, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.grid()
    plt.close()
    au, nau = 0, 0
    return fig, au, nau
