"""Package initializer for srcs.

Exports convenience names so modules in this folder can be imported from the project root:

  from srcs.preprocessing_functions import vif_prune
  # or
  from srcs import vif_prune

"""

from .preprocessing_functions import vif_prune, compute_vif_from_corr, bic_grid, best_gmm_by_bic, compute_stat_summary
from .plot_functions import plot_bic_and_clusters, plot_feature_scatter

__all__ = ["vif_prune", "compute_vif_from_corr", "bic_grid", "best_gmm_by_bic", "plot_bic_and_clusters", "plot_feature_scatter", "compute_stat_summary"]
