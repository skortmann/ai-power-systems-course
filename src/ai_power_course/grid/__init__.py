"""Power-system tooling for the capstone.

``networks``  the catalogue of pandapower grids and the pretrain/held-out split
``sampling``  generating many operating points across many grids
``graphs``    converting a solved network into GNN tensors
``physics``   power-balance residuals, voltage and loading violations
"""

from __future__ import annotations

__all__ = ["graphs", "networks", "physics", "sampling"]
