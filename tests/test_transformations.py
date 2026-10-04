from copy import deepcopy
from inspect import signature
from pathlib import Path

import numpy as np
import pytest

import irspecflow
import irspecflow.transformations
from irspecflow import Spectrum
from irspecflow.io import read_spectrum
from irspecflow.transformations import (
    interpolate_spectrum,
    transmittance_to_absorbance,
)


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _transmittance_spectrum(
    axis=(3000.0, 2000.0, 1000.0),
    response=(100.0, 10.0, 1.0),
    **kwargs,
):
    return Spectrum(
        axis,
        response,
        response_type="transmittance",
        response_unit="%",
        **kwargs,
    )


def _numeric_spectrum(
    axis=(1000.0, 2000.0, 3000.0),
    response=(10.0, 20.0, 30.0),
    **kwargs,
):
    return Spectrum(
        axis,
        response,
        **kwargs,
    )


def test_transformations_public_interface():
    assert (
        irspecflow.transformations.transmittance_to_absorbance
        is transmittance_to_absorbance
    )
    assert (
        irspecflow.transformations.interpolate_spectrum
        is interpolate_spectrum
    )
    assert irspecflow.transformations.__all__ == [
        "transmittance_to_absorbance",
        "interpolate_spectrum",
    ]
    assert not hasattr(irspecflow, "transmittance_to_absorbance")
    assert not hasattr(irspecflow, "interpolate_spectrum")


def test_interpolation_does_not_expose_method_argument():
    parameters = signature(interpolate_spectrum).parameters

    assert list(parameters) == [
        "spectrum",
        "target_axis",
    ]


@pytest.mark.parametrize(
    "operation",
    [
        lambda spectrum: transmittance_to_absorbance(spectrum),
        lambda spectrum: interpolate_spectrum(spectrum, [1500.0]),
    ],
)
def test_transformations_accept_subclasses_and_return_plain_spectrum(
    operation,
):
    class DerivedSpectrum(Spectrum):
        pass

    spectrum = DerivedSpectrum(
        [1000.0, 2000.0],
        [100.0, 10.0],
        response_type="transmittance",
        response_unit="%",
    )

    result = operation(spectrum)

    assert type(result) is Spectrum
    assert result is not spectrum


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ([100.0], [0.0]),
        ([10.0], [1.0]),
        ([100.0, 10.0, 1.0], [0.0, 1.0, 2.0]),
    ],
)
def test_transmittance_to_absorbance_reference_values(
    response,
    expected,
):
    axis = list(range(len(response)))
    spectrum = _transmittance_spectrum(
        axis=axis,
        response=response,
    )

    result = transmittance_to_absorbance(spectrum)

    np.testing.assert_allclose(
        result.response,
        expected,
    )


def test_transmittance_to_absorbance_accepts_value_above_100_percent():
    spectrum = _transmittance_spectrum(
        axis=[1000.0],
        response=[125.0],
    )

    result = transmittance_to_absorbance(spectrum)

    np.testing.assert_allclose(
        result.response,
        [-np.log10(1.25)],
    )
    assert result.response[0] < 0.0


@pytest.mark.parametrize(
    "spectrum",
    [
        None,
        [100.0, 10.0],
        {"response": [100.0, 10.0]},
    ],
)
def test_transmittance_to_absorbance_rejects_unsupported_input(spectrum):
    with pytest.raises(
        TypeError,
        match="spectrum must be a Spectrum",
    ):
        transmittance_to_absorbance(spectrum)


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_transmittance_to_absorbance_rejects_nonpositive_values(value):
    spectrum = _transmittance_spectrum(
        axis=[1000.0],
        response=[value],
    )

    with pytest.raises(
        ValueError,
        match="greater than 0",
    ):
        transmittance_to_absorbance(spectrum)


@pytest.mark.parametrize(
    ("value", "code"),
    [
        (np.nan, "response_nan"),
        (np.inf, "response_positive_infinity"),
        (-np.inf, "response_negative_infinity"),
    ],
)
def test_transmittance_to_absorbance_rejects_nonfinite_response(
    value,
    code,
):
    spectrum = _transmittance_spectrum(
        axis=[1000.0],
        response=[value],
    )

    with pytest.raises(
        ValueError,
        match=code,
    ):
        transmittance_to_absorbance(spectrum)


def test_transmittance_to_absorbance_rejects_empty_spectrum():
    spectrum = _transmittance_spectrum(
        axis=[],
        response=[],
    )

    with pytest.raises(
        ValueError,
        match="empty_spectrum",
    ):
        transmittance_to_absorbance(spectrum)


@pytest.mark.parametrize(
    ("response_type", "response_unit", "message"),
    [
        (None, "%", "response type"),
        ("absorbance", None, "response type"),
        ("absorbance", "%", "inconsistent_response_unit"),
        ("reflectance", "%", "unrecognized_response_type"),
        ("transmittance", None, "response unit"),
        ("transmittance", "percent", "unrecognized_response_unit"),
    ],
)
def test_transmittance_to_absorbance_requires_exact_declarations(
    response_type,
    response_unit,
    message,
):
    spectrum = Spectrum(
        [1000.0],
        [50.0],
        response_type=response_type,
        response_unit=response_unit,
    )

    with pytest.raises(
        ValueError,
        match=message,
    ):
        transmittance_to_absorbance(spectrum)


@pytest.mark.parametrize(
    "axis",
    [
        [3000.0, np.nan, 1000.0],
        [3000.0, np.inf, 1000.0],
        [3000.0, -np.inf, 1000.0],
        [3000.0, 2000.0, 2000.0],
        [3000.0, 1000.0, 2000.0],
    ],
)
def test_axis_only_findings_do_not_block_response_conversion(axis):
    spectrum = _transmittance_spectrum(axis=axis)

    result = transmittance_to_absorbance(spectrum)

    np.testing.assert_array_equal(
        result.axis,
        spectrum.axis,
    )
    np.testing.assert_allclose(
        result.response,
        [0.0, 1.0, 2.0],
    )


def test_unrecognized_axis_unit_does_not_block_response_conversion():
    spectrum = _transmittance_spectrum(
        axis_unit="cm-1",
    )

    result = transmittance_to_absorbance(spectrum)

    np.testing.assert_allclose(
        result.response,
        [0.0, 1.0, 2.0],
    )
    assert result.axis_unit == "cm-1"


def test_conversion_preserves_source_and_sets_output_contract():
    spectrum = _transmittance_spectrum(
        axis=[2000.0, 1000.0],
        response=[80.0, 40.0],
        axis_unit="cm^-1",
        spectrum_id="spectrum-a",
        hierarchy={
            "sample": {
                "id": "sample-a",
            }
        },
        metadata={
            "acquisition": {
                "resolution_cm1": 4,
            },
            "provenance": {
                "import": {
                    "importer": "example",
                },
                "operator": "analyst-a",
            },
        },
    )
    original_axis = spectrum.axis.copy()
    original_response = spectrum.response.copy()
    original_hierarchy = deepcopy(spectrum.hierarchy)
    original_metadata = deepcopy(spectrum.metadata)

    result = transmittance_to_absorbance(spectrum)

    np.testing.assert_array_equal(spectrum.axis, original_axis)
    np.testing.assert_array_equal(spectrum.response, original_response)
    assert spectrum.hierarchy == original_hierarchy
    assert spectrum.metadata == original_metadata
    assert spectrum.axis_unit == "cm^-1"
    assert spectrum.response_type == "transmittance"
    assert spectrum.response_unit == "%"
    assert spectrum.spectrum_id == "spectrum-a"
    assert not spectrum.axis.flags.writeable
    assert not spectrum.response.flags.writeable

    assert result.axis_unit == "cm^-1"
    assert result.response_type == "absorbance"
    assert result.response_unit is None
    assert result.spectrum_id == "spectrum-a"
    assert result.hierarchy == original_hierarchy
    assert result.metadata["acquisition"] == original_metadata["acquisition"]
    assert (
        result.metadata["provenance"]["import"]
        == original_metadata["provenance"]["import"]
    )
    assert result.metadata["provenance"]["operator"] == "analyst-a"
    assert result.metadata["provenance"]["transformations"] == [
        {
            "operation": (
                "irspecflow.transformations."
                "transmittance_to_absorbance"
            ),
        }
    ]
    assert result.axis.dtype == np.float64
    assert result.response.dtype == np.float64
    assert not result.axis.flags.writeable
    assert not result.response.flags.writeable


def test_conversion_creates_transformation_provenance():
    spectrum = _transmittance_spectrum(
        axis=[1000.0],
        response=[50.0],
    )

    result = transmittance_to_absorbance(spectrum)

    assert result.metadata == {
        "provenance": {
            "transformations": [
                {
                    "operation": (
                        "irspecflow.transformations."
                        "transmittance_to_absorbance"
                    ),
                }
            ],
        }
    }
    assert spectrum.metadata == {}


def test_conversion_appends_existing_transformation_provenance_in_order():
    previous_entry = {
        "operation": "example.previous_operation",
        "parameter": {
            "value": 1,
        },
    }
    spectrum = _transmittance_spectrum(
        axis=[1000.0],
        response=[50.0],
        metadata={
            "provenance": {
                "transformations": [previous_entry],
            }
        },
    )
    original_metadata = deepcopy(spectrum.metadata)

    result = transmittance_to_absorbance(spectrum)

    assert result.metadata["provenance"]["transformations"] == [
        previous_entry,
        {
            "operation": (
                "irspecflow.transformations."
                "transmittance_to_absorbance"
            ),
        },
    ]
    assert spectrum.metadata == original_metadata


@pytest.mark.parametrize(
    "metadata",
    [
        {"provenance": "manual"},
        {"provenance": {"transformations": {}}},
    ],
)
def test_conversion_rejects_malformed_provenance_containers(metadata):
    spectrum = _transmittance_spectrum(
        axis=[1000.0],
        response=[50.0],
        metadata=metadata,
    )

    with pytest.raises(
        ValueError,
        match="provenance",
    ):
        transmittance_to_absorbance(spectrum)


@pytest.mark.parametrize(
    ("source_axis", "target_axis", "expected_response"),
    [
        (
            [1000.0, 2000.0, 3000.0],
            [1000.0, 1500.0, 3000.0],
            [10.0, 15.0, 30.0],
        ),
        (
            [1000.0, 2000.0, 3000.0],
            [3000.0, 1500.0, 1000.0],
            [30.0, 15.0, 10.0],
        ),
        (
            [3000.0, 2000.0, 1000.0],
            [1000.0, 1500.0, 3000.0],
            [10.0, 15.0, 30.0],
        ),
        (
            [3000.0, 2000.0, 1000.0],
            [3000.0, 1500.0, 1000.0],
            [30.0, 15.0, 10.0],
        ),
    ],
)
def test_interpolation_supports_source_and_target_directions(
    source_axis,
    target_axis,
    expected_response,
):
    spectrum = _numeric_spectrum(
        axis=source_axis,
        response=[value / 100.0 for value in source_axis],
    )

    result = interpolate_spectrum(spectrum, target_axis)

    np.testing.assert_array_equal(result.axis, target_axis)
    np.testing.assert_allclose(result.response, expected_response)


def test_interpolation_accepts_one_point_target():
    spectrum = _numeric_spectrum(
        axis=[1000.0, 2000.0],
        response=[10.0, 20.0],
    )

    result = interpolate_spectrum(spectrum, [1500.0])

    np.testing.assert_array_equal(result.axis, [1500.0])
    np.testing.assert_allclose(result.response, [15.0])


def test_interpolation_rejects_empty_source():
    spectrum = _numeric_spectrum(
        axis=[],
        response=[],
    )

    with pytest.raises(
        ValueError,
        match="empty_spectrum",
    ):
        interpolate_spectrum(spectrum, [1000.0])


def test_interpolation_rejects_one_point_source():
    spectrum = _numeric_spectrum(
        axis=[1000.0],
        response=[10.0],
    )

    with pytest.raises(
        ValueError,
        match="At least two spectral points",
    ):
        interpolate_spectrum(spectrum, [1000.0])


def test_interpolation_accepts_endpoints_and_calculates_interior_values():
    spectrum = _numeric_spectrum(
        response=[1.0, 3.0, 7.0],
    )

    result = interpolate_spectrum(
        spectrum,
        [1000.0, 1250.0, 2500.0, 3000.0],
    )

    np.testing.assert_allclose(
        result.response,
        [1.0, 1.5, 5.0, 7.0],
    )


@pytest.mark.parametrize(
    "spectrum",
    [
        None,
        [1.0, 2.0],
        {"axis": [1.0, 2.0]},
    ],
)
def test_interpolation_rejects_unsupported_source_input(spectrum):
    with pytest.raises(
        TypeError,
        match="spectrum must be a Spectrum",
    ):
        interpolate_spectrum(spectrum, [1.5])


def test_interpolation_rejects_empty_target():
    with pytest.raises(
        ValueError,
        match="at least one value",
    ):
        interpolate_spectrum(_numeric_spectrum(), [])


@pytest.mark.parametrize(
    "target_axis",
    [
        ["1000", "not-numeric"],
        [1000.0 + 1.0j, 2000.0 + 0.0j],
        {"axis": [1000.0, 2000.0]},
    ],
)
def test_interpolation_rejects_unsupported_or_nonnumeric_target(
    target_axis,
):
    with pytest.raises(TypeError):
        interpolate_spectrum(
            _numeric_spectrum(),
            target_axis,
        )


def test_interpolation_rejects_scalar_target():
    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        interpolate_spectrum(
            _numeric_spectrum(),
            1500.0,
        )


def test_interpolation_rejects_multidimensional_target():
    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        interpolate_spectrum(
            _numeric_spectrum(),
            [[1000.0, 1500.0]],
        )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_interpolation_rejects_nonfinite_target(value):
    with pytest.raises(
        ValueError,
        match="finite",
    ):
        interpolate_spectrum(
            _numeric_spectrum(),
            [1000.0, value],
        )


def test_interpolation_rejects_duplicate_target():
    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        interpolate_spectrum(
            _numeric_spectrum(),
            [1000.0, 1000.0],
        )


def test_interpolation_rejects_duplicates_after_float64_preparation():
    base = 2**53
    spectrum = _numeric_spectrum(
        axis=[base, base + 2],
        response=[10.0, 20.0],
    )

    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        interpolate_spectrum(
            spectrum,
            [base, base + 1],
        )


def test_interpolation_rejects_nonmonotonic_target():
    with pytest.raises(
        ValueError,
        match="strictly increasing or strictly decreasing",
    ):
        interpolate_spectrum(
            _numeric_spectrum(),
            [1000.0, 2500.0, 2000.0],
        )


@pytest.mark.parametrize("target_axis", [[999.0], [3001.0]])
def test_interpolation_rejects_out_of_range_target(target_axis):
    with pytest.raises(
        ValueError,
        match="closed numerical range",
    ):
        interpolate_spectrum(_numeric_spectrum(), target_axis)


@pytest.mark.parametrize(
    ("axis", "code"),
    [
        ([1000.0, np.nan, 3000.0], "axis_nan"),
        ([1000.0, np.inf, 3000.0], "axis_positive_infinity"),
        ([1000.0, -np.inf, 3000.0], "axis_negative_infinity"),
    ],
)
def test_interpolation_rejects_nonfinite_source_axis(axis, code):
    spectrum = _numeric_spectrum(axis=axis)

    with pytest.raises(
        ValueError,
        match=code,
    ):
        interpolate_spectrum(spectrum, [1500.0])


@pytest.mark.parametrize(
    ("response", "code"),
    [
        ([10.0, np.nan, 30.0], "response_nan"),
        ([10.0, np.inf, 30.0], "response_positive_infinity"),
        ([10.0, -np.inf, 30.0], "response_negative_infinity"),
    ],
)
def test_interpolation_rejects_nonfinite_source_response(
    response,
    code,
):
    spectrum = _numeric_spectrum(response=response)

    with pytest.raises(
        ValueError,
        match=code,
    ):
        interpolate_spectrum(spectrum, [1500.0])


@pytest.mark.parametrize(
    ("axis", "code"),
    [
        ([1000.0, 2000.0, 2000.0], "duplicate_axis_values"),
        ([1000.0, 3000.0, 2000.0], "non_monotonic_axis"),
    ],
)
def test_interpolation_rejects_invalid_source_axis_structure(axis, code):
    spectrum = _numeric_spectrum(axis=axis)

    with pytest.raises(
        ValueError,
        match=code,
    ):
        interpolate_spectrum(spectrum, [1500.0])


@pytest.mark.parametrize(
    ("axis_unit", "response_type", "response_unit"),
    [
        ("cm-1", "reflectance", "percent"),
        ("cm^-1", "absorbance", "%"),
    ],
)
def test_declaration_findings_do_not_block_interpolation(
    axis_unit,
    response_type,
    response_unit,
):
    spectrum = _numeric_spectrum(
        axis_unit=axis_unit,
        response_type=response_type,
        response_unit=response_unit,
    )

    result = interpolate_spectrum(
        spectrum,
        [1500.0, 2500.0],
    )

    np.testing.assert_allclose(result.response, [15.0, 25.0])
    assert result.axis_unit == axis_unit
    assert result.response_type == response_type
    assert result.response_unit == response_unit


def test_identity_target_returns_distinct_spectrum_and_records_operation():
    spectrum = _numeric_spectrum(
        axis=[3000.0, 2000.0, 1000.0],
        response=[30.0, 20.0, 10.0],
    )

    result = interpolate_spectrum(spectrum, spectrum.axis)

    assert result is not spectrum
    np.testing.assert_array_equal(result.axis, spectrum.axis)
    np.testing.assert_array_equal(result.response, spectrum.response)
    assert result.metadata["provenance"]["transformations"] == [
        {
            "operation": (
                "irspecflow.transformations."
                "interpolate_spectrum"
            ),
            "method": "linear",
        }
    ]


def test_interpolation_preserves_source_and_output_contract():
    spectrum = _numeric_spectrum(
        axis=[3000.0, 2000.0, 1000.0],
        response=[0.30, 0.20, 0.10],
        axis_unit="cm^-1",
        response_type="absorbance",
        spectrum_id="spectrum-a",
        hierarchy={
            "sample": {
                "id": "sample-a",
            }
        },
        metadata={
            "acquisition": {
                "resolution_cm1": 4,
            },
            "provenance": {
                "import": {
                    "importer": "example",
                },
                "operator": "analyst-a",
            },
        },
    )
    original_axis = spectrum.axis.copy()
    original_response = spectrum.response.copy()
    original_hierarchy = deepcopy(spectrum.hierarchy)
    original_metadata = deepcopy(spectrum.metadata)

    result = interpolate_spectrum(
        spectrum,
        [3000.0, 2500.0, 1000.0],
    )

    np.testing.assert_array_equal(spectrum.axis, original_axis)
    np.testing.assert_array_equal(spectrum.response, original_response)
    assert spectrum.hierarchy == original_hierarchy
    assert spectrum.metadata == original_metadata
    assert spectrum.axis_unit == "cm^-1"
    assert spectrum.response_type == "absorbance"
    assert spectrum.response_unit is None
    assert spectrum.spectrum_id == "spectrum-a"
    assert not spectrum.axis.flags.writeable
    assert not spectrum.response.flags.writeable

    assert result.axis_unit == "cm^-1"
    assert result.response_type == "absorbance"
    assert result.response_unit is None
    assert result.spectrum_id == "spectrum-a"
    assert result.hierarchy == original_hierarchy
    assert result.metadata["acquisition"] == original_metadata["acquisition"]
    assert (
        result.metadata["provenance"]["import"]
        == original_metadata["provenance"]["import"]
    )
    assert result.metadata["provenance"]["operator"] == "analyst-a"
    assert result.metadata["provenance"]["transformations"] == [
        {
            "operation": (
                "irspecflow.transformations."
                "interpolate_spectrum"
            ),
            "method": "linear",
        }
    ]
    assert result.axis.dtype == np.float64
    assert result.response.dtype == np.float64
    assert not result.axis.flags.writeable
    assert not result.response.flags.writeable


def test_interpolation_appends_existing_transformation_provenance_in_order():
    previous_entry = {
        "operation": "example.previous_operation",
        "parameter": 1,
    }
    spectrum = _numeric_spectrum(
        axis=[1000.0, 2000.0],
        response=[10.0, 20.0],
        metadata={
            "provenance": {
                "transformations": [previous_entry],
            }
        },
    )
    original_metadata = deepcopy(spectrum.metadata)

    result = interpolate_spectrum(spectrum, [1500.0])

    assert result.metadata["provenance"]["transformations"] == [
        previous_entry,
        {
            "operation": (
                "irspecflow.transformations."
                "interpolate_spectrum"
            ),
            "method": "linear",
        },
    ]
    assert spectrum.metadata == original_metadata


@pytest.mark.parametrize(
    "metadata",
    [
        {"provenance": "manual"},
        {"provenance": {"transformations": {}}},
    ],
)
def test_interpolation_rejects_malformed_provenance_containers(metadata):
    spectrum = _numeric_spectrum(
        axis=[1000.0, 2000.0],
        response=[10.0, 20.0],
        metadata=metadata,
    )

    with pytest.raises(
        ValueError,
        match="provenance",
    ):
        interpolate_spectrum(spectrum, [1500.0])


def test_conversion_then_interpolation_records_transformation_order():
    spectrum = _transmittance_spectrum()

    converted = transmittance_to_absorbance(spectrum)
    interpolated = interpolate_spectrum(
        converted,
        [3000.0, 2500.0, 1000.0],
    )

    assert interpolated.metadata["provenance"]["transformations"] == [
        {
            "operation": (
                "irspecflow.transformations."
                "transmittance_to_absorbance"
            ),
        },
        {
            "operation": (
                "irspecflow.transformations."
                "interpolate_spectrum"
            ),
            "method": "linear",
        },
    ]


def test_interpolation_preserves_import_provenance():
    spectrum = read_spectrum(
        DATA_DIR / "example_spectrum.csv",
        axis_column="wavenumber",
        response_column="absorbance",
        axis_unit="cm^-1",
        response_type="absorbance",
    )
    import_record = deepcopy(
        spectrum.metadata["provenance"]["import"]
    )

    result = interpolate_spectrum(
        spectrum,
        [4000.0, 2500.0, 400.0],
    )

    assert result.metadata["provenance"]["import"] == import_record
    assert spectrum.metadata["provenance"]["import"] == import_record


def test_imported_transmittance_spectrum_can_be_converted(tmp_path):
    path = tmp_path / "transmittance.csv"
    path.write_text(
        "axis,response\n"
        "3000,100\n"
        "2000,10\n"
        "1000,1\n",
        encoding="utf-8",
    )

    spectrum = read_spectrum(
        path,
        axis_column="axis",
        response_column="response",
        axis_unit="cm^-1",
        response_type="transmittance",
        response_unit="%",
    )
    import_record = deepcopy(
        spectrum.metadata["provenance"]["import"]
    )

    result = transmittance_to_absorbance(spectrum)

    np.testing.assert_allclose(result.response, [0.0, 1.0, 2.0])
    assert result.metadata["provenance"]["import"] == import_record