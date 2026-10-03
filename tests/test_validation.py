from copy import deepcopy
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import numpy as np
import pytest

import irspecflow
import irspecflow.validation
from irspecflow import Spectrum
from irspecflow.io import read_spectrum
from irspecflow.validation import ValidationFinding, validate_spectrum


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_validation_public_interface():
    assert irspecflow.validation.ValidationFinding is ValidationFinding
    assert irspecflow.validation.validate_spectrum is validate_spectrum
    assert irspecflow.validation.__all__ == [
        "ValidationFinding",
        "validate_spectrum",
    ]


def test_validation_is_not_exposed_at_package_level():
    assert not hasattr(irspecflow, "ValidationFinding")
    assert not hasattr(irspecflow, "validate_spectrum")


def test_validation_finding_has_two_value_fields():
    finding = ValidationFinding(
        code="example_code",
        message="Example message.",
    )

    assert [field.name for field in fields(ValidationFinding)] == [
        "code",
        "message",
    ]
    assert finding == ValidationFinding(
        code="example_code",
        message="Example message.",
    )


def test_validation_finding_is_immutable():
    finding = ValidationFinding(
        code="example_code",
        message="Example message.",
    )

    with pytest.raises(FrozenInstanceError):
        setattr(finding, "code", "changed_code")


def test_validate_spectrum_returns_tuple_for_clean_spectrum():
    spectrum = Spectrum(
        [4000.0, 3000.0, 2000.0],
        [0.10, 0.20, 0.30],
        axis_unit="cm^-1",
        response_type="absorbance",
    )

    findings = validate_spectrum(spectrum)

    assert findings == ()
    assert isinstance(findings, tuple)


@pytest.mark.parametrize(
    "value",
    [
        None,
        [4000.0, 3000.0],
        {"axis": [4000.0, 3000.0]},
    ],
)
def test_validate_spectrum_rejects_unsupported_input(value):
    with pytest.raises(
        TypeError,
        match="spectrum must be a Spectrum",
    ):
        validate_spectrum(value)


def test_validate_spectrum_accepts_spectrum_subclass():
    class DerivedSpectrum(Spectrum):
        pass

    spectrum = DerivedSpectrum(
        [4000.0, 3000.0],
        [0.10, 0.20],
    )

    assert validate_spectrum(spectrum) == ()


def test_validate_spectrum_reports_empty_spectrum():
    spectrum = Spectrum([], [])

    assert validate_spectrum(spectrum) == (
        ValidationFinding(
            code="empty_spectrum",
            message="Spectrum contains no spectral points.",
        ),
    )


def test_validate_spectrum_checks_declarations_for_empty_spectrum():
    spectrum = Spectrum(
        [],
        [],
        axis_unit="cm-1",
        response_type="Absorbance",
        response_unit="percent",
    )

    assert [finding.code for finding in validate_spectrum(spectrum)] == [
        "empty_spectrum",
        "unrecognized_axis_unit",
        "unrecognized_response_type",
        "unrecognized_response_unit",
    ]


def test_validate_spectrum_accepts_one_finite_point():
    spectrum = Spectrum(
        [1000.0],
        [0.10],
    )

    assert validate_spectrum(spectrum) == ()


@pytest.mark.parametrize(
    "axis",
    [
        [1000.0, 2000.0],
        [2000.0, 1000.0],
    ],
)
def test_validate_spectrum_accepts_two_distinct_finite_points(axis):
    spectrum = Spectrum(
        axis,
        [0.10, 0.20],
    )

    assert validate_spectrum(spectrum) == ()


def test_validate_spectrum_reports_equal_two_point_axis_as_duplicate_only():
    spectrum = Spectrum(
        [1000.0, 1000.0],
        [0.10, 0.20],
    )

    assert validate_spectrum(spectrum) == (
        ValidationFinding(
            code="duplicate_axis_values",
            message="Axis contains duplicate finite values.",
        ),
    )


@pytest.mark.parametrize(
    "axis",
    [
        [1000.0, 2000.0, 3000.0, 4000.0],
        [4000.0, 3000.0, 2000.0, 1000.0],
    ],
)
def test_validate_spectrum_accepts_strictly_monotonic_axis(axis):
    spectrum = Spectrum(
        axis,
        [0.10, 0.20, 0.30, 0.40],
    )

    assert validate_spectrum(spectrum) == ()


@pytest.mark.parametrize(
    ("value", "code", "message"),
    [
        (
            np.nan,
            "axis_nan",
            "Axis contains one or more NaN values.",
        ),
        (
            np.inf,
            "axis_positive_infinity",
            "Axis contains one or more positive infinity values.",
        ),
        (
            -np.inf,
            "axis_negative_infinity",
            "Axis contains one or more negative infinity values.",
        ),
    ],
)
def test_validate_spectrum_reports_axis_nonfinite_conditions(
    value,
    code,
    message,
):
    spectrum = Spectrum(
        [4000.0, value, 2000.0],
        [0.10, 0.20, 0.30],
    )

    assert validate_spectrum(spectrum) == (
        ValidationFinding(
            code=code,
            message=message,
        ),
    )


@pytest.mark.parametrize(
    ("value", "code", "message"),
    [
        (
            np.nan,
            "response_nan",
            "Response contains one or more NaN values.",
        ),
        (
            np.inf,
            "response_positive_infinity",
            "Response contains one or more positive infinity values.",
        ),
        (
            -np.inf,
            "response_negative_infinity",
            "Response contains one or more negative infinity values.",
        ),
    ],
)
def test_validate_spectrum_reports_response_nonfinite_conditions(
    value,
    code,
    message,
):
    spectrum = Spectrum(
        [4000.0, 3000.0, 2000.0],
        [0.10, value, 0.30],
    )

    assert validate_spectrum(spectrum) == (
        ValidationFinding(
            code=code,
            message=message,
        ),
    )


def test_validate_spectrum_reports_each_nonfinite_category_once_in_order():
    spectrum = Spectrum(
        [
            np.nan,
            np.inf,
            -np.inf,
            4000.0,
            3000.0,
            2000.0,
        ],
        [
            np.nan,
            np.inf,
            -np.inf,
            0.10,
            0.20,
            0.30,
        ],
    )

    assert [finding.code for finding in validate_spectrum(spectrum)] == [
        "axis_nan",
        "axis_positive_infinity",
        "axis_negative_infinity",
        "response_nan",
        "response_positive_infinity",
        "response_negative_infinity",
    ]


def test_validate_spectrum_reports_repeated_nonfinite_condition_once():
    spectrum = Spectrum(
        [4000.0, np.nan, np.nan, 2000.0],
        [0.10, 0.20, 0.30, 0.40],
    )

    assert [finding.code for finding in validate_spectrum(spectrum)] == [
        "axis_nan",
    ]


def test_response_nonfinite_values_do_not_suppress_axis_structure_checks():
    spectrum = Spectrum(
        [4000.0, 2000.0, 3000.0],
        [0.10, np.nan, 0.30],
    )

    assert [finding.code for finding in validate_spectrum(spectrum)] == [
        "response_nan",
        "non_monotonic_axis",
    ]


def test_validate_spectrum_reports_adjacent_duplicate_axis_values():
    spectrum = Spectrum(
        [4000.0, 3000.0, 3000.0, 2000.0],
        [0.10, 0.20, 0.30, 0.40],
    )

    assert validate_spectrum(spectrum) == (
        ValidationFinding(
            code="duplicate_axis_values",
            message="Axis contains duplicate finite values.",
        ),
    )


def test_validate_spectrum_reports_nonadjacent_duplicate_axis_values():
    spectrum = Spectrum(
        [4000.0, 3000.0, 2000.0, 3000.0],
        [0.10, 0.20, 0.30, 0.40],
    )

    assert [finding.code for finding in validate_spectrum(spectrum)] == [
        "duplicate_axis_values",
        "non_monotonic_axis",
    ]


def test_validate_spectrum_does_not_treat_near_equal_values_as_duplicates():
    next_value = np.nextafter(2000.0, 3000.0)
    spectrum = Spectrum(
        [1000.0, 2000.0, next_value, 3000.0],
        [0.10, 0.20, 0.30, 0.40],
    )

    assert validate_spectrum(spectrum) == ()


def test_validate_spectrum_reports_genuine_axis_reversal():
    spectrum = Spectrum(
        [4000.0, 2000.0, 3000.0],
        [0.10, 0.20, 0.30],
    )

    assert validate_spectrum(spectrum) == (
        ValidationFinding(
            code="non_monotonic_axis",
            message=(
                "Axis reverses direction and is not consistently increasing "
                "or decreasing."
            ),
        ),
    )


def test_validate_spectrum_reports_duplicate_and_reversal_together():
    spectrum = Spectrum(
        [4000.0, 3000.0, 3000.0, 3500.0],
        [0.10, 0.20, 0.30, 0.40],
    )

    assert [finding.code for finding in validate_spectrum(spectrum)] == [
        "duplicate_axis_values",
        "non_monotonic_axis",
    ]


def test_validate_spectrum_detects_finite_duplicates_with_nonfinite_axis():
    spectrum = Spectrum(
        [4000.0, np.nan, 3000.0, 3000.0, 2000.0],
        [0.10, 0.20, 0.30, 0.40, 0.50],
    )

    assert [finding.code for finding in validate_spectrum(spectrum)] == [
        "axis_nan",
        "duplicate_axis_values",
    ]


@pytest.mark.parametrize(
    ("value", "code"),
    [
        (np.nan, "axis_nan"),
        (np.inf, "axis_positive_infinity"),
        (-np.inf, "axis_negative_infinity"),
    ],
)
def test_nonfinite_axis_suppresses_monotonicity_finding(
    value,
    code,
):
    spectrum = Spectrum(
        [4000.0, value, 2000.0, 3000.0],
        [0.10, 0.20, 0.30, 0.40],
    )

    assert [finding.code for finding in validate_spectrum(spectrum)] == [
        code,
    ]


def test_validate_spectrum_accepts_missing_declarations():
    spectrum = Spectrum(
        [4000.0, 3000.0],
        [0.10, 0.20],
    )

    assert validate_spectrum(spectrum) == ()


@pytest.mark.parametrize(
    "response_type",
    [
        "absorbance",
        "transmittance",
    ],
)
def test_validate_spectrum_accepts_recognized_response_types(response_type):
    spectrum = Spectrum(
        [4000.0, 3000.0],
        [0.10, 0.20],
        axis_unit="cm^-1",
        response_type=response_type,
    )

    assert validate_spectrum(spectrum) == ()


def test_validate_spectrum_accepts_percent_without_response_type():
    spectrum = Spectrum(
        [4000.0, 3000.0],
        [80.0, 75.0],
        axis_unit="cm^-1",
        response_unit="%",
    )

    assert validate_spectrum(spectrum) == ()


@pytest.mark.parametrize(
    "axis_unit",
    [
        "cm-1",
        "cm⁻¹",
        "1/cm",
        "CM^-1",
        " cm^-1",
        "cm^-1 ",
    ],
)
def test_validate_spectrum_does_not_normalize_axis_unit(axis_unit):
    spectrum = Spectrum(
        [4000.0, 3000.0],
        [0.10, 0.20],
        axis_unit=axis_unit,
    )

    assert validate_spectrum(spectrum) == (
        ValidationFinding(
            code="unrecognized_axis_unit",
            message=(
                f"Unrecognized axis unit {axis_unit!r}; "
                "the recognized axis unit is 'cm^-1'."
            ),
        ),
    )


@pytest.mark.parametrize(
    "response_type",
    [
        "Absorbance",
        "TRANSMITTANCE",
        " absorbance",
        "transmittance ",
    ],
)
def test_validate_spectrum_does_not_normalize_response_type(
    response_type,
):
    spectrum = Spectrum(
        [4000.0, 3000.0],
        [0.10, 0.20],
        response_type=response_type,
    )

    assert validate_spectrum(spectrum) == (
        ValidationFinding(
            code="unrecognized_response_type",
            message=(
                f"Unrecognized response type {response_type!r}; "
                "the recognized response types are 'absorbance' and "
                "'transmittance'."
            ),
        ),
    )


@pytest.mark.parametrize(
    "response_unit",
    [
        "percent",
        "Percent",
        " %",
        "% ",
    ],
)
def test_validate_spectrum_does_not_normalize_response_unit(
    response_unit,
):
    spectrum = Spectrum(
        [4000.0, 3000.0],
        [80.0, 75.0],
        response_unit=response_unit,
    )

    assert validate_spectrum(spectrum) == (
        ValidationFinding(
            code="unrecognized_response_unit",
            message=(
                f"Unrecognized response unit {response_unit!r}; "
                "the recognized supplied response unit is '%'."
            ),
        ),
    )


def test_validate_spectrum_accepts_transmittance_with_percent():
    spectrum = Spectrum(
        [4000.0, 3000.0],
        [80.0, 75.0],
        response_type="transmittance",
        response_unit="%",
    )

    assert validate_spectrum(spectrum) == ()


def test_validate_spectrum_reports_absorbance_with_percent():
    spectrum = Spectrum(
        [4000.0, 3000.0],
        [0.10, 0.20],
        response_type="absorbance",
        response_unit="%",
    )

    assert validate_spectrum(spectrum) == (
        ValidationFinding(
            code="inconsistent_response_unit",
            message=(
                "Response unit '%' is inconsistent with declared response "
                "type 'absorbance'."
            ),
        ),
    )


@pytest.mark.parametrize(
    ("response_type", "response_unit", "expected_codes"),
    [
        (
            "absorbance",
            "percent",
            ["unrecognized_response_unit"],
        ),
        (
            "reflectance",
            "%",
            ["unrecognized_response_type"],
        ),
        (
            "reflectance",
            "arbitrary",
            [
                "unrecognized_response_type",
                "unrecognized_response_unit",
            ],
        ),
    ],
)
def test_validate_spectrum_does_not_infer_consistency_for_unknown_declarations(
    response_type,
    response_unit,
    expected_codes,
):
    spectrum = Spectrum(
        [4000.0, 3000.0],
        [0.10, 0.20],
        response_type=response_type,
        response_unit=response_unit,
    )

    assert [
        finding.code
        for finding in validate_spectrum(spectrum)
    ] == expected_codes


def test_validate_spectrum_returns_complex_findings_in_contract_order():
    spectrum = Spectrum(
        [4000.0, 3000.0, 3000.0, 3500.0, 2000.0],
        [np.nan, np.inf, -np.inf, 0.40, 0.50],
        axis_unit="cm-1",
        response_type="Absorbance",
        response_unit="percent",
    )

    assert validate_spectrum(spectrum) == (
        ValidationFinding(
            code="response_nan",
            message="Response contains one or more NaN values.",
        ),
        ValidationFinding(
            code="response_positive_infinity",
            message="Response contains one or more positive infinity values.",
        ),
        ValidationFinding(
            code="response_negative_infinity",
            message="Response contains one or more negative infinity values.",
        ),
        ValidationFinding(
            code="duplicate_axis_values",
            message="Axis contains duplicate finite values.",
        ),
        ValidationFinding(
            code="non_monotonic_axis",
            message=(
                "Axis reverses direction and is not consistently increasing "
                "or decreasing."
            ),
        ),
        ValidationFinding(
            code="unrecognized_axis_unit",
            message=(
                "Unrecognized axis unit 'cm-1'; "
                "the recognized axis unit is 'cm^-1'."
            ),
        ),
        ValidationFinding(
            code="unrecognized_response_type",
            message=(
                "Unrecognized response type 'Absorbance'; "
                "the recognized response types are 'absorbance' and "
                "'transmittance'."
            ),
        ),
        ValidationFinding(
            code="unrecognized_response_unit",
            message=(
                "Unrecognized response unit 'percent'; "
                "the recognized supplied response unit is '%'."
            ),
        ),
    )


def test_validate_spectrum_accepts_existing_imported_example():
    spectrum = read_spectrum(
        DATA_DIR / "example_spectrum.csv",
        axis_column="wavenumber",
        response_column="absorbance",
        axis_unit="cm^-1",
        response_type="absorbance",
    )

    assert validate_spectrum(spectrum) == ()


def test_validate_spectrum_reports_imported_structural_condition(
    tmp_path,
):
    path = tmp_path / "duplicate_axis.csv"
    path.write_text(
        "axis,response\n"
        "4000,0.10\n"
        "3000,0.20\n"
        "3000,0.30\n"
        "2000,0.40\n",
        encoding="utf-8",
    )

    spectrum = read_spectrum(
        path,
        axis_column="axis",
        response_column="response",
        axis_unit="cm^-1",
        response_type="absorbance",
    )

    np.testing.assert_array_equal(
        spectrum.axis,
        [4000.0, 3000.0, 3000.0, 2000.0],
    )
    assert validate_spectrum(spectrum) == (
        ValidationFinding(
            code="duplicate_axis_values",
            message="Axis contains duplicate finite values.",
        ),
    )


def test_validate_spectrum_does_not_mutate_direct_spectrum():
    spectrum = Spectrum(
        [4000.0, 3000.0, 3000.0, 3500.0],
        [0.10, np.nan, 0.30, 0.40],
        axis_unit="cm-1",
        response_type="Absorbance",
        response_unit="percent",
        spectrum_id="sample-a-replicate-1",
        hierarchy={
            "sample": {
                "id": "sample-a",
            },
            "replicate": 1,
        },
        metadata={
            "acquisition": {
                "resolution_cm1": 4,
                "scans": 32,
            },
            "notes": {
                "operator": "analyst-a",
            },
        },
    )

    axis_before = spectrum.axis.copy()
    response_before = spectrum.response.copy()
    axis_dtype_before = spectrum.axis.dtype
    response_dtype_before = spectrum.response.dtype
    hierarchy_before = deepcopy(spectrum.hierarchy)
    metadata_before = deepcopy(spectrum.metadata)

    validate_spectrum(spectrum)

    np.testing.assert_array_equal(
        spectrum.axis,
        axis_before,
    )
    np.testing.assert_array_equal(
        spectrum.response,
        response_before,
    )
    assert spectrum.axis.dtype == axis_dtype_before
    assert spectrum.response.dtype == response_dtype_before
    assert not spectrum.axis.flags.writeable
    assert not spectrum.response.flags.writeable
    assert spectrum.axis_unit == "cm-1"
    assert spectrum.response_type == "Absorbance"
    assert spectrum.response_unit == "percent"
    assert spectrum.spectrum_id == "sample-a-replicate-1"
    assert spectrum.hierarchy == hierarchy_before
    assert spectrum.metadata == metadata_before


def test_validate_spectrum_does_not_mutate_imported_spectrum():
    spectrum = read_spectrum(
        DATA_DIR / "example_spectrum.csv",
        axis_column="wavenumber",
        response_column="absorbance",
        axis_unit="cm^-1",
        response_type="absorbance",
        spectrum_id="example-spectrum",
        hierarchy={
            "source": {
                "id": "source-a",
            }
        },
        metadata={
            "acquisition": {
                "resolution_cm1": 4,
            },
            "notes": {
                "operator": "analyst-a",
            },
        },
    )

    axis_before = spectrum.axis.copy()
    response_before = spectrum.response.copy()
    hierarchy_before = deepcopy(spectrum.hierarchy)
    metadata_before = deepcopy(spectrum.metadata)

    validate_spectrum(spectrum)

    np.testing.assert_array_equal(
        spectrum.axis,
        axis_before,
    )
    np.testing.assert_array_equal(
        spectrum.response,
        response_before,
    )
    assert not spectrum.axis.flags.writeable
    assert not spectrum.response.flags.writeable
    assert spectrum.axis_unit == "cm^-1"
    assert spectrum.response_type == "absorbance"
    assert spectrum.response_unit is None
    assert spectrum.spectrum_id == "example-spectrum"
    assert spectrum.hierarchy == hierarchy_before
    assert spectrum.metadata == metadata_before
    assert (
        spectrum.metadata["provenance"]["import"]
        == metadata_before["provenance"]["import"]
    )