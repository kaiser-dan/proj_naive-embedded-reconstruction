# --- Standard library ---

# --- Test utility ---
import pytest

# --- Source code ---
from embmplxrec.remnants import observer

# ========== Fixtures ==========
# --- Test Fixtures ---

# --- Test helpers ---
def _fmt_deg_seq(g): return list(dict(g.degree()).values())
def _get_edges(g): return set(g.edges())


# ========== Test suite ==========
