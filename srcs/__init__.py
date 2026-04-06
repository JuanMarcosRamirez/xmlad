"""Package initializer for srcs.

Exports convenience names so modules in this folder can be imported from the project root:

  from srcs.preprocessing_functions import vif_prune
  # or
  from srcs import vif_prune

"""

from .preprocessing_functions import vif_prune, compute_vif_from_corr, bic_grid, best_gmm_by_bic

__all__ = ["vif_prune", "compute_vif_from_corr", "bic_grid", "best_gmm_by_bic"]
