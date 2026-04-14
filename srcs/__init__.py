"""Package initializer for srcs.

Exports convenience names so modules in this folder can be imported from the project root:

  from srcs.preprocessing_functions import vif_prune
  # or
  from srcs import vif_prune

"""

from .preprocessing_functions import vif_prune, bic_grid, best_gmm_by_bic, compute_stat_summary
from .plot_functions import plot_bic_and_clusters, plot_feature_scatter, plot_distribution_pair, format_axis, plot_outlier_detection, plot_pruning
from .misc import stat_summary
from .anmdet import detect_outliers_iqr, detect_outliers_zscore, detect_outliers_mad
from .models import model_based_clustering, GMMclustering, build_decision_tree

__all__ = ["vif_prune", 
           "bic_grid", 
           "best_gmm_by_bic", 
           "plot_bic_and_clusters", 
           "plot_feature_scatter", 
           "plot_outlier_detection",
           "compute_stat_summary", 
           "stat_summary", 
           "plot_distribution_pair", 
           "detect_outliers_iqr",
           "detect_outliers_zscore",
           "detect_outliers_mad",
           "format_axis",
           "model_based_clustering",
           "GMMclustering",
           "build_decision_tree",
           "plot_pruning"]