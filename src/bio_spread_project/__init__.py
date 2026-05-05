"""BioSpread: a standalone plasmid geographic spread early-warning project."""

import warnings

# Third-party deprecation noise that does not affect project behavior and
# cannot be fixed from this codebase directly.
warnings.filterwarnings(
    "ignore",
    message="`torch_geometric.distributed` has been deprecated*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message="`torch.jit.script` is deprecated*",
    category=DeprecationWarning,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
]
