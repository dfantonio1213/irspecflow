import numpy as np
import pytest

import irspecflow
from irspecflow import Spectrum


def test_spectrum_is_exposed_at_package_level():
    assert irspecflow.Spectrum is Spectrum


def test_spectrum_accepts_array_like_input_and_normalizes_dtype():
    spectrum = Spectrum(
        [4000, 3000, 2000],
        (0.10, 0.20, 0.30),
    )

    np.testing.assert_array_equal(
        spectrum.axis,
        [4000.0, 3000.0, 2000.0],
    )
    np.testing.assert_array_equal(
        spectrum.response,
        [0.10, 0.20, 0.30],
    )

    assert spectrum.axis.dtype == np.float64
    assert spectrum.response.dtype == np.float64


def test_spectrum_copies_numerical_inputs():
    axis = np.array([4000.0, 3000.0])
    response = np.array([0.10, 0.20])

    spectrum = Spectrum(axis, response)

    axis[0] = 1000.0
    response[0] = 9.99

    np.testing.assert_array_equal(
        spectrum.axis,
        [4000.0, 3000.0],
    )
    np.testing.assert_array_equal(
        spectrum.response,
        [0.10, 0.20],
    )


def test_spectrum_numerical_data_are_read_only():
    spectrum = Spectrum(
        [4000.0, 3000.0],
        [0.10, 0.20],
    )

    assert not spectrum.axis.flags.writeable
    assert not spectrum.response.flags.writeable

    with pytest.raises(ValueError):
        spectrum.axis[0] = 1000.0

    with pytest.raises(ValueError):
        spectrum.response.setflags(write=True)


def test_spectrum_stores_optional_descriptive_fields():
    spectrum = Spectrum(
        [4000.0, 3000.0],
        [75.0, 70.0],
        axis_unit="cm^-1",
        response_type="transmittance",
        response_unit="percent",
        spectrum_id="sample-a-replicate-1",
        hierarchy={
            "sample": "sample-a",
            "replicate": 1,
        },
        metadata={
            "acquisition": {
                "resolution_cm1": 4,
                "scans": 32,
            }
        },
    )

    assert spectrum.axis_unit == "cm^-1"
    assert spectrum.response_type == "transmittance"
    assert spectrum.response_unit == "percent"
    assert spectrum.spectrum_id == "sample-a-replicate-1"

    assert spectrum.hierarchy == {
        "sample": "sample-a",
        "replicate": 1,
    }

    assert spectrum.metadata == {
        "acquisition": {
            "resolution_cm1": 4,
            "scans": 32,
        }
    }


def test_spectrum_descriptive_fields_default_to_unset_or_empty():
    spectrum = Spectrum(
        [4000.0],
        [0.10],
    )

    assert spectrum.axis_unit is None
    assert spectrum.response_type is None
    assert spectrum.response_unit is None
    assert spectrum.spectrum_id is None
    assert spectrum.hierarchy == {}
    assert spectrum.metadata == {}


def test_spectrum_deep_copies_hierarchy_and_metadata():
    hierarchy = {
        "sample": {
            "id": "sample-a",
        }
    }

    metadata = {
        "acquisition": {
            "resolution_cm1": 4,
        }
    }

    spectrum = Spectrum(
        [4000.0],
        [0.10],
        hierarchy=hierarchy,
        metadata=metadata,
    )

    hierarchy["sample"]["id"] = "changed"
    metadata["acquisition"]["resolution_cm1"] = 8

    assert spectrum.hierarchy["sample"]["id"] == "sample-a"
    assert spectrum.metadata["acquisition"]["resolution_cm1"] == 4


def test_spectrum_hierarchy_and_metadata_remain_editable():
    spectrum = Spectrum(
        [4000.0],
        [0.10],
    )

    spectrum.hierarchy["sample"] = "sample-a"
    spectrum.metadata["note"] = "checked"

    assert spectrum.hierarchy["sample"] == "sample-a"
    assert spectrum.metadata["note"] == "checked"


@pytest.mark.parametrize(
    ("axis", "response", "field"),
    [
        (4000.0, [0.10], "axis"),
        ([4000.0], 0.10, "response"),
    ],
)
def test_spectrum_rejects_scalar_numerical_input(
    axis,
    response,
    field,
):
    with pytest.raises(
        ValueError,
        match=rf"{field} must be one-dimensional",
    ):
        Spectrum(axis, response)


@pytest.mark.parametrize(
    ("axis", "response", "field"),
    [
        (
            [[4000.0, 3000.0]],
            [0.10, 0.20],
            "axis",
        ),
        (
            [4000.0, 3000.0],
            [[0.10, 0.20]],
            "response",
        ),
    ],
)
def test_spectrum_rejects_multidimensional_input(
    axis,
    response,
    field,
):
    with pytest.raises(
        ValueError,
        match=rf"{field} must be one-dimensional",
    ):
        Spectrum(axis, response)


def test_spectrum_requires_matching_axis_and_response_lengths():
    with pytest.raises(
        ValueError,
        match="same length",
    ):
        Spectrum(
            [4000.0, 3000.0],
            [0.10],
        )


@pytest.mark.parametrize(
    ("axis", "response", "field"),
    [
        (
            np.array([4000.0 + 1.0j]),
            [0.10],
            "axis",
        ),
        (
            [4000.0],
            np.array([0.10 + 1.0j]),
            "response",
        ),
    ],
)
def test_spectrum_rejects_complex_input(
    axis,
    response,
    field,
):
    with pytest.raises(
        TypeError,
        match=rf"{field} must contain real numeric values",
    ):
        Spectrum(axis, response)


@pytest.mark.parametrize(
    ("axis", "response", "field"),
    [
        (
            ["not-numeric"],
            [0.10],
            "axis",
        ),
        (
            [4000.0],
            ["not-numeric"],
            "response",
        ),
    ],
)
def test_spectrum_rejects_non_numeric_input(
    axis,
    response,
    field,
):
    with pytest.raises(
        TypeError,
        match=rf"{field} must contain real numeric values",
    ):
        Spectrum(axis, response)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"axis_unit": 1},
        {"response_type": 1},
        {"response_unit": 1},
        {"spectrum_id": 1},
    ],
)
def test_spectrum_requires_string_descriptors_or_none(kwargs):
    with pytest.raises(
        TypeError,
        match="must be a string or None",
    ):
        Spectrum(
            [4000.0],
            [0.10],
            **kwargs,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "hierarchy": [
                ("sample", "sample-a"),
            ]
        },
        {
            "metadata": [
                ("note", "checked"),
            ]
        },
    ],
)
def test_spectrum_requires_mapping_metadata(kwargs):
    with pytest.raises(
        TypeError,
        match="must be a mapping or None",
    ):
        Spectrum(
            [4000.0],
            [0.10],
            **kwargs,
        )


def test_spectrum_preserves_supplied_axis_order():
    spectrum = Spectrum(
        [4000.0, 2000.0, 3000.0],
        [0.10, 0.20, 0.30],
    )

    np.testing.assert_array_equal(
        spectrum.axis,
        [4000.0, 2000.0, 3000.0],
    )


@pytest.mark.parametrize(
    ("axis", "response"),
    [
        (
            [],
            [],
        ),
        (
            [4000.0, 4000.0],
            [0.10, 0.20],
        ),
        (
            [4000.0, 2000.0, 3000.0],
            [0.10, 0.20, 0.30],
        ),
        (
            [4000.0, 3000.0],
            [np.nan, np.inf],
        ),
    ],
)
def test_spectrum_defers_broader_structural_validation(
    axis,
    response,
):
    Spectrum(axis, response)