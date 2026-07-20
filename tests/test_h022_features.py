from __future__ import annotations

import numpy as np

from sphinx_trace.h022_features import (
    H022_TREE_FEATURE_WIDTH,
    H022_WALLET_START,
    H022_WALLET_STOP,
    assemble_tree_features,
    component_folds,
    wallet_ablation,
)


def _features(rows: int = 5) -> np.ndarray:
    return assemble_tree_features(
        np.zeros((rows, 128), dtype=np.float32),
        np.linspace(-1.0, 1.0, rows, dtype=np.float32),
        np.zeros(rows, dtype=np.float32),
        np.zeros((rows, 9), dtype=np.float32),
        np.zeros((rows, 7), dtype=np.float32),
        np.zeros((rows, 3), dtype=np.float32),
        np.zeros((rows, 3), dtype=np.float32),
        np.tile(np.asarray([[0.4, 0.6, 2.4, 1.6, 0.4, 0.6]], dtype=np.float32), (rows, 1)),
        np.arange(rows, dtype=np.int64) % 2,
    )


def test_h022_features_are_finite_and_have_registered_width() -> None:
    features = _features()
    assert features.shape == (5, H022_TREE_FEATURE_WIDTH)
    assert np.isfinite(features).all()


def test_component_fold_never_splits_a_component() -> None:
    components = np.asarray([7, 7, 11, 11, 11, 19], dtype=np.int64)
    folds = component_folds(components, 3, 17)
    assert folds[0] == folds[1]
    assert folds[2] == folds[3] == folds[4]
    assert ((folds >= 0) & (folds < 3)).all()


def test_wallet_ablation_does_not_touch_other_groups() -> None:
    features = _features(6)
    features[:, H022_WALLET_START:H022_WALLET_STOP] = np.arange(
        6, dtype=np.float32
    )[:, None]
    zero = wallet_ablation(features, "zero", seed=17)
    shuffled = wallet_ablation(features, "shuffle", seed=17)
    assert not zero[:, H022_WALLET_START:H022_WALLET_STOP].any()
    assert np.array_equal(zero[:, :H022_WALLET_START], features[:, :H022_WALLET_START])
    assert np.array_equal(shuffled[:, :H022_WALLET_START], features[:, :H022_WALLET_START])
    assert sorted(shuffled[:, H022_WALLET_START].tolist()) == list(range(6))
