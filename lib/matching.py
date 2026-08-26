"""Cell barcode matching utilities used across all benchmark levels.

Core concept: (16mer, lane) compound key — the only reliable way to
match cells across pipeline outputs and reference data when multiple
10x lanes share overlapping 16mer barcode spaces.

All levels (GEX, extraction, assignment) use this module for consistent
cell matching.
"""

import re
from typing import Dict, List, Tuple, Set, Optional

# Regex: 16mer barcode with lane suffix (-LNN or -N format)
_BC_LANE = re.compile(r'^([ACGT]{16})-L(\d+)$')
_BC_GT   = re.compile(r'^[^_]*_([ACGT]{16})(?:-\d+)?$')


def extract_16mer(bc: str) -> str:
    """Extract 16mer barcode from a full barcode string.

    Handles:
      - Plain 16mer:       'AAACCCAAGAAACCAA'
      - Lane-suffixed:     'AAACCCAAGAAACCAA-L01'
      - Ground truth:      'l1_AAACCCAAGAAACCAA'
      - Raw with -1:       'AAACCCAAGAAACCAA-1'
    """
    m = _BC_LANE.match(bc)
    if m:
        return m.group(1)
    # Fallback: strip any non-ACGT prefix/suffix
    m2 = re.search(r'([ACGT]{16})', bc)
    if m2:
        return m2.group(1)
    return bc


def extract_lane_id(bc: str) -> int:
    """Extract lane number from a barcode string.

    Handles:
      - Lane-suffixed:     'AAACCCAAGAAACCAA-L01' → 1
      - Ground truth:      'AAACCCAAGAAACCAA-1'   → 1
      - No lane info:       'AAACCCAAGAAACCAA'     → 0 (unknown)
    """
    m = re.search(r'-L(\d+)$', bc)
    if m:
        return int(m.group(1))
    m = re.search(r'-(\d+)$', bc)
    if m:
        return int(m.group(1))
    # Try prefix format: l1_16mer
    m = _BC_GT.match(bc)
    if m:
        return 0  # no lane info in pure 16mer
    return 0


def make_compound_key(bc: str, lane: Optional[int] = None) -> Tuple[str, int]:
    """Build (16mer, lane) compound key from a barcode string.

    If lane is provided explicitly, uses that value (e.g. from reference
    gem_group metadata). Otherwise parses the barcode suffix.
    """
    m16 = extract_16mer(bc)
    if lane is not None:
        return (m16, lane)
    return (m16, extract_lane_id(bc))


def build_compound_key_index(
    barcodes: List[str],
    gem_groups: Optional[List[int]] = None,
) -> Dict[Tuple[str, int], int]:
    """Build a {(16mer, lane): array_index} lookup dict.

    Args:
        barcodes: List of full barcode strings.
        gem_groups: Optional parallel list of gem_group integers
                    (from h5ad/h5mu obs metadata). If provided,
                    used as lane identifier. Otherwise parsed from
                    barcode suffix.

    Returns:
        Dict mapping (16mer, lane) → position index in original list.
        Keys are unique — duplicate (16mer, lane) pairs would
        indicate corrupted data.
    """
    index: Dict[Tuple[str, int], int] = {}
    for i, bc in enumerate(barcodes):
        m16 = extract_16mer(bc)
        if gem_groups is not None:
            lane = int(gem_groups[i])
        else:
            lane = extract_lane_id(bc)
        key = (m16, lane)
        index[key] = i  # last-wins for duplicates
    return index


def build_16mer_index(
    barcodes: List[str],
) -> Dict[str, int]:
    """Build a {16mer: array_index} lookup dict (no lane dimension).

    Only use for single-lane datasets where lane collisions
    are impossible (e.g. Papalexi 2021 single 10x lane).
    For multi-lane datasets always prefer build_compound_key_index().
    """
    index: Dict[str, int] = {}
    for i, bc in enumerate(barcodes):
        m16 = extract_16mer(bc)
        index[m16] = i
    return index
