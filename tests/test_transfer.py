import numpy as np
import pytest

from vtk2cesium.transfer import (
    NonFinitePolicy,
    ScalarMapping,
    ScalarPreprocessConfig,
    TransferFunction,
    TransferPoint,
    preprocess_scalar,
)


def test_identity_preserves_shape_order_and_finite_values() -> None:
    values = np.array([[[0.0, 1.0], [np.nan, 3.0]]], dtype=np.float64)
    result = preprocess_scalar(values)

    assert result.values.shape == values.shape
    assert np.array_equal(result.values.ravel(), [0.0, 1.0, 0.0, 3.0])
    assert np.array_equal(result.mask.ravel(), [True, True, False, True])
    assert result.source_range == (0.0, 3.0)
    assert not result.values.flags.writeable
    assert not result.mask.flags.writeable


def test_linear_mapping_clips_to_unit_interval() -> None:
    values = np.array([-5.0, 0.0, 5.0, 10.0, 15.0])
    result = preprocess_scalar(
        values,
        ScalarPreprocessConfig(
            mapping=ScalarMapping.LINEAR,
            source_range=(0.0, 10.0),
        ),
    )

    assert result.values == pytest.approx([0.0, 0.0, 0.5, 1.0, 1.0])
    assert result.output_range == (0.0, 1.0)


def test_log_mapping_and_constant_field_have_defined_outputs() -> None:
    logarithmic = preprocess_scalar(
        np.array([0.0, 9.0, 99.0]),
        ScalarPreprocessConfig(
            mapping=ScalarMapping.LOG,
            source_range=(0.0, 99.0),
            log_base=10.0,
        ),
    )
    constant = preprocess_scalar(
        np.full((2, 2, 2), 7.0),
        ScalarPreprocessConfig(mapping=ScalarMapping.LINEAR),
    )

    assert logarithmic.values == pytest.approx([0.0, 0.2596373, 1.0], abs=1e-6)
    assert np.array_equal(constant.values, np.zeros((2, 2, 2), dtype=np.float32))


def test_threshold_masks_values_without_reordering() -> None:
    values = np.arange(6, dtype=np.float32).reshape((1, 2, 3))
    result = preprocess_scalar(
        values,
        ScalarPreprocessConfig(threshold=(2.0, 4.0), fill_value=-1.0),
    )

    assert np.array_equal(result.values.ravel(), [-1.0, -1.0, 2.0, 3.0, 4.0, -1.0])
    assert np.array_equal(result.mask.ravel(), [False, False, True, True, True, False])


def test_non_finite_policies_are_explicit() -> None:
    values = np.array([1.0, np.nan])
    filled = preprocess_scalar(
        values,
        ScalarPreprocessConfig(non_finite=NonFinitePolicy.FILL, fill_value=-2.0),
    )

    assert np.array_equal(filled.values, [1.0, -2.0])
    assert filled.mask.all()
    with pytest.raises(ValueError, match="NaN or infinity"):
        preprocess_scalar(
            values,
            ScalarPreprocessConfig(non_finite=NonFinitePolicy.ERROR),
        )


def test_piecewise_mapping_interpolates_control_points() -> None:
    result = preprocess_scalar(
        np.array([0.0, 2.5, 5.0, 7.5, 10.0]),
        ScalarPreprocessConfig(
            mapping=ScalarMapping.PIECEWISE,
            piecewise_points=((0.0, 0.0), (5.0, 1.0), (10.0, 0.0)),
        ),
    )

    assert result.values == pytest.approx([0.0, 0.5, 1.0, 0.5, 0.0])


def test_transfer_function_samples_rgba_and_builds_lookup_table() -> None:
    transfer = TransferFunction(
        (
            TransferPoint(0.0, (0.0, 0.0, 1.0), 0.1),
            TransferPoint(1.0, (1.0, 0.0, 0.0), 0.9),
        )
    )

    sampled = transfer.sample(np.array([0.0, 0.5, 1.0]))
    table = transfer.lookup_table(4)

    assert sampled.shape == (3, 4)
    assert sampled[0] == pytest.approx((0.0, 0.0, 1.0, 0.1))
    assert sampled[1] == pytest.approx((0.5, 0.0, 0.5, 0.5))
    assert sampled[2] == pytest.approx((1.0, 0.0, 0.0, 0.9))
    assert table.shape == (4, 4)


def test_transfer_function_rejects_unsorted_points() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        TransferFunction(
            (
                TransferPoint(1.0, (1.0, 0.0, 0.0), 1.0),
                TransferPoint(0.0, (0.0, 0.0, 1.0), 0.0),
            )
        )
