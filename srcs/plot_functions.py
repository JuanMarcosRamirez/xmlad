import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from srcs.anmdet import (
    detect_outliers_iqr,
    detect_outliers_zscore,
    detect_outliers_mad,
)

def plot_bic_and_clusters(bic_ite: np.ndarray, 
                          iterations: int, 
                          max_num_clusters: int, 
                          k_opt: int, 
                          x: np.ndarray, 
                          clusters: np.ndarray) -> None:
    meanbic = np.mean(bic_ite, axis=0)
    minbic = np.min(meanbic)
    maxbic = np.max(meanbic)
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    for ii in range(iterations):
        if (ii ==0):
            ax[0].plot(np.arange(1, max_num_clusters), bic_ite[ii,:], lw=2, c='lightblue', label='Realization')
        else:
            ax[0].plot(np.arange(1, max_num_clusters), bic_ite[ii,:], lw=2, c='lightblue')
    ax[0].plot(np.arange(1, max_num_clusters), meanbic, '-o',lw=2, c='darkblue', label='Average')
    ax[0].plot([k_opt, k_opt], [minbic, maxbic],'k--', lw=2)
    ax[0].set_xlabel('Number of clusters', fontsize=14)
    ax[0].set_ylabel('BIC', fontsize=14)
    # ax[0].set_title('(a)', fontsize=16)
    ax[0].tick_params(axis='x', labelsize=12)
    ax[0].tick_params(axis='y', labelsize=12)
    ax[0].set_ylim([minbic-20, maxbic])
    ax[0].set_xlim([0, 40])
    # plt.text(num_opt_clusters_bic + 2, np.mean(bic_ite, axis=0)[num_opt_clusters_bic-1]-20, str(num_opt_clusters_bic) + ' clusters', fontsize=14)
    ax[0].legend(fontsize=12, loc='lower right')


    for i in range(len(np.unique(clusters))):
        ax[1].scatter(x[clusters == i], np.zeros(len(x[clusters == i])), s=100)
    ax[1].set_xlabel('1 - |corr|', fontsize=14)
    # ax[1].title('(b)', fontsize=16)
    ax[1].tick_params(axis='x', labelsize=12)
    ax[1].tick_params(axis='y', labelsize=12)
    fig.tight_layout()
    # # plt.savefig('data/figures/MonteCarloSim.png', dpi=200, bbox_inches='tight' )
    plt.show()
    



def plot_feature_scatter(out_num: pd.DataFrame, kpi_aux: np.ndarray, cols: pd.Index, ncols: int = 5) -> None:
    if (len(cols)%ncols) == 0 :
        nrows = len(cols)//ncols
    else:
        nrows = len(cols)//ncols + 1
    fig, ax = plt.subplots(nrows, ncols, figsize=(4.00*ncols,2.50*nrows), tight_layout=True)
    for ii in range(len(cols)):
        ax[ii//ncols][ii%ncols].plot(out_num[cols[ii]], kpi_aux, 'o', color='tab:red')
        ax[ii//ncols][ii%ncols].set_xlabel(cols[ii])
        ax[ii//ncols][ii%ncols].set_ylabel('KPI (kbps)')
    fig.tight_layout()
    plt.show()
    
    # -------------------------------------------------------------------
# Plot helpers
# -------------------------------------------------------------------
def format_axis(ax, xlabel, ylabel, title, tick_fs=8, label_fs=8, title_fs=8):
    ax.set_xlabel(xlabel, fontsize=label_fs)
    ax.set_ylabel(ylabel, fontsize=label_fs)
    ax.set_title(title, fontsize=title_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.xaxis.get_offset_text().set_fontsize(tick_fs)

def plot_distribution_pair(raw: pd.Series, log_s: pd.Series, bins: int = 25) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6, 3), tight_layout=True)

    configs = [
        (raw.dropna(), "KPI Histogram", "Value", "Frequency", "tab:blue", 10, 8, 8),
        (log_s.dropna(), "Log10 KPI Histogram", "Value", "Frequency", "tab:orange", 10, 8, 8),
    ]
    for ax, (s, title, xlabel, ylabel, color, title_fs, label_fs, tick_fs) in zip(axes, configs):
        ax.hist(s, bins=bins, color=color, edgecolor="black")
        format_axis(ax, xlabel, ylabel, title, tick_fs=tick_fs, label_fs=label_fs, title_fs=title_fs)
    plt.show()


DETECTORS = {
    "IQR": detect_outliers_iqr,
    "Z-Score": detect_outliers_zscore,
    "MAD": detect_outliers_mad,
}
    
def plot_outlier_detection(s: pd.Series, xlabel: str, suffix: str = "", bins: int = 25) -> dict[str, pd.Series]:
    s = s.dropna()
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), tight_layout=True)

    masks = {}
    for ax, (name, detector) in zip(axes, DETECTORS.items()):
        mask = detector(s)
        masks[name] = mask

        ax.hist(
            [s[~mask], s[mask]],
            bins=bins,
            stacked=True,
            color=["tab:blue", "tab:orange"],
            edgecolor="black",
            label=["Inliers", "Outliers"],
        )
        format_axis(
            ax,
            xlabel=xlabel,
            ylabel="Counts",
            title=f"{name}",
            tick_fs=8,
            label_fs=8,
            title_fs=10,
        )
        ax.legend(fontsize=8)
        ax.autoscale(axis="both", tight=True)

    plt.show()
    return masks

def plot_pruning(ccp_alphas, scores):
    fig = plt.figure(dpi=100)
    ax = fig.add_subplot()
    plt.plot(ccp_alphas, scores, marker='o', drawstyle="steps-post", color='tab:blue',linewidth=2)
    plt.xlabel(r"$\alpha$", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2.0)  # change width
    plt.show()