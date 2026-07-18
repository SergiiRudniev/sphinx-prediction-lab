from __future__ import annotations

from pathlib import Path

import numpy as np
from scripts.qualify_h011_feature_pack import _atomic_numpy, _link_or_copy


def test_qualified_pack_helpers_are_atomic_and_reuse_immutable_files(tmp_path: Path) -> None:
    source = tmp_path / "source.npy"
    target = tmp_path / "nested" / "target.npy"
    values = np.array([1, 0, 1], dtype=np.uint8)
    _atomic_numpy(source, values)
    method = _link_or_copy(source, target)
    assert method in {"hardlink", "copy"}
    assert np.array_equal(np.load(target), values)
    assert _link_or_copy(source, target) == "existing"
