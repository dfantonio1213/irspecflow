from copy import deepcopy
from inspect import Parameter, signature

import numpy as np
import pytest

import irspecflow
import irspecflow.qc
from irspecflow import Spectrum
from irspecflow.qc import response_range


def _spectrum(
    axis=(1000.0, 2000.0, 3000.0),
    response=(1.0, 2.0, 3.0),
    **kwargs,
):
    return Spectrum(
        axis,
        response,
        **kwargs,
    )


def test_qc_public_interface():
    assert irspecflow.qc.response_range is response_range
    assert irspecflow.qc.__all__ == ["response_range"]
    assert not hasattr(
        irspecflow,
        "response_range",
    )


def test_response_range_public_signature():
    parameters = signature(
        response_range
    ).parameters

    assert list(parameters) == ["spectrum"]
    assert (
        parameters["spectrum"].kind
        is Parameter.POSITIONAL_OR_KEYWORD
    )
    assert parameters["spectrum"].default is Parameter.empty


def test_response_range_accepts_spectrum_subclass():
    class DerivedSpectrum(Spectrum):
        pass

    spectrum = DerivedSpectrum(
        [1000.0, 2000.0],
        [2.0, 5.0],
    )

    assert response_range(spectrum) == 3.0


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ([1.0, 4.0, 2.0], 3.0),
        ([-5.0, -2.0, -8.0], 6.0),
        ([-3.0, 4.0, -1.0], 7.0),
        ([3.0, 1.0, 2.0], 2.0),
        ([5.0, 5.0, 5.0], 0.0),
        ([5.0], 0.0),
    ],
)
def test_response_range_reference_values(
    response,
    expected,
):
    spectrum = _spectrum(
        axis=tuple(
            range(len(response))
        ),
        response=response,
    )

    result = response_range(spectrum)

    assert result == expected
    assert type(result) is float


@pytest.mark.parametrize(
    "spectrum",
    [
        None,
        [1.0, 2.0],
        {
            "response": [
                1.0,
                2.0,
            ]
        },
    ],
)
def test_response_range_rejects_unsupported_input(
    spectrum,
):
    with pytest.raises(
        TypeError,
        match="spectrum must be a Spectrum",
    ):
        response_range(spectrum)


def test_response_range_rejects_empty_spectrum():
    with pytest.raises(
        ValueError,
        match="empty_spectrum",
    ):
        response_range(
            Spectrum([], [])
        )


@pytest.mark.parametrize(
    ("value", "code"),
    [
        (
            np.nan,
            "response_nan",
        ),
        (
            np.inf,
            "response_positive_infinity",
        ),
        (
            -np.inf,
            "response_negative_infinity",
        ),
    ],
)
def test_response_range_rejects_nonfinite_response(
    value,
    code,
):
    spectrum = _spectrum(
        response=(
            1.0,
            value,
            3.0,
        ),
    )

    with pytest.raises(
        ValueError,
        match=code,
    ):
        response_range(spectrum)


@pytest.mark.parametrize(
    "axis",
    [
        (
            1000.0,
            np.nan,
            3000.0,
        ),
        (
            1000.0,
            np.inf,
            3000.0,
        ),
        (
            1000.0,
            -np.inf,
            3000.0,
        ),
        (
            1000.0,
            2000.0,
            2000.0,
        ),
        (
            1000.0,
            3000.0,
            2000.0,
        ),
    ],
)
def test_axis_only_findings_do_not_block_response_range(
    axis,
):
    spectrum = _spectrum(
        axis=axis,
        response=(
            2.0,
            5.0,
            3.0,
        ),
    )

    assert response_range(spectrum) == 3.0


@pytest.mark.parametrize(
    "declarations",
    [
        {
            "axis_unit": "cm-1",
        },
        {
            "response_type": "Reflectance",
        },
        {
            "response_unit": "percent",
        },
        {
            "response_type": "absorbance",
            "response_unit": "%",
        },
        {
            "axis_unit": "cm-1",
            "response_type": "Reflectance",
            "response_unit": "percent",
        },
    ],
)
def test_scientific_declaration_findings_do_not_block_response_range(
    declarations,
):
    spectrum = _spectrum(
        response=(
            2.0,
            5.0,
            3.0,
        ),
        **declarations,
    )

    assert response_range(spectrum) == 3.0


def test_response_range_does_not_mutate_source_state():
    spectrum = _spectrum(
        axis=(
            3000.0,
            1000.0,
            2000.0,
        ),
        response=(
            0.30,
            -0.10,
            0.20,
        ),
        axis_unit="cm-1",
        response_type="Absorbance",
        response_unit="percent",
        spectrum_id="spectrum-a",
        hierarchy={
            "source": {
                "id": "source-a",
            },
            "replicate": 1,
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
                "transformations": [
                    {
                        "operation": "example.operation",
                    }
                ],
            },
        },
    )

    axis_before = spectrum.axis.copy()
    response_before = spectrum.response.copy()
    axis_dtype_before = spectrum.axis.dtype
    response_dtype_before = spectrum.response.dtype
    hierarchy_before = deepcopy(
        spectrum.hierarchy
    )
    metadata_before = deepcopy(
        spectrum.metadata
    )

    result = response_range(spectrum)

    assert result == 0.4
    assert type(result) is float

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

    assert spectrum.axis_unit == "cm-1"
    assert spectrum.response_type == "Absorbance"
    assert spectrum.response_unit == "percent"
    assert spectrum.spectrum_id == "spectrum-a"
    assert spectrum.hierarchy == hierarchy_before
    assert spectrum.metadata == metadata_before

    assert (
        spectrum.metadata["provenance"]["import"]
        == metadata_before["provenance"]["import"]
    )
    assert (
        spectrum.metadata[
            "provenance"
        ]["transformations"]
        == metadata_before[
            "provenance"
        ]["transformations"]
    )

    assert "qc" not in spectrum.metadata
    assert not spectrum.axis.flags.writeable
    assert not spectrum.response.flags.writeable