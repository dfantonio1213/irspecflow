"""Explicit transformation utilities for infrared spectra."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .spectrum import Spectrum
from .validation import ValidationFinding, validate_spectrum


__all__ = ["transmittance_to_absorbance", "interpolate_spectrum"]


_CONVERSION_BLOCKING_CODES = {
    "empty_spectrum",
    "response_nan",
    "response_positive_infinity",
    "response_negative_infinity",
    "unrecognized_response_type",
    "unrecognized_response_unit",
    "inconsistent_response_unit",
}

_INTERPOLATION_BLOCKING_CODES = {
    "empty_spectrum",
    "axis_nan",
    "axis_positive_infinity",
    "axis_negative_infinity",
    "response_nan",
    "response_positive_infinity",
    "response_negative_infinity",
    "duplicate_axis_values",
    "non_monotonic_axis",
}


def transmittance_to_absorbance(
    spectrum: Spectrum,
) -> Spectrum:
    """Convert one percent-transmittance Spectrum to absorbance.

    Parameters
    ----------
    spectrum : Spectrum
        Spectrum containing percent-transmittance response values. Subclasses of
        ``Spectrum`` are also accepted. ``response_type`` must be exactly
        ``"transmittance"`` and ``response_unit`` must be exactly ``"%"``.

    Returns
    -------
    Spectrum
        New Spectrum containing the converted absorbance response. The result is
        an ordinary ``Spectrum`` even when the input is a subclass. The returned
        spectrum has ``response_type="absorbance"`` and ``response_unit=None``.
        The spectral axis, axis unit, spectrum identifier, hierarchy, and
        unrelated metadata are preserved.

    Raises
    ------
    TypeError
        If ``spectrum`` is not a ``Spectrum`` instance.
    ValueError
        If the spectrum is empty, the required response declarations are missing
        or incompatible, response values are non-finite or not greater than zero,
        or the existing provenance metadata has an incompatible structure.

    Notes
    -----
    Absorbance is calculated as ``A = -log10(T / 100)``, where ``T`` is percent
    transmittance. Values above 100 %T are mathematically valid and may produce
    negative absorbance; this function does not treat that condition as a
    measurement-quality failure.

    The supplied spectrum is not modified. Existing metadata and provenance are
    preserved, and one transformation record is appended to
    ``metadata["provenance"]["transformations"]``.

    Structural validation is applied only to conditions needed for this response
    conversion. Conditions limited to the spectral axis do not prevent conversion
    because the calculation uses only the response values. This function does not
    sort or repair the spectral axis, perform axis-unit conversion, interpolate,
    perform quality control, or apply spectral preprocessing.
    """

    findings = validate_spectrum(spectrum)
    blocking_findings = _blocking_findings(
        findings,
        _CONVERSION_BLOCKING_CODES,
    )

    if blocking_findings:
        details = "; ".join(
            f"{finding.message} ({finding.code})"
            for finding in blocking_findings
        )

        raise ValueError(
            "Structural validation reported conditions that prevent "
            "transmittance-to-absorbance conversion: "
            f"{details}"
        )

    if spectrum.response_type != "transmittance":
        raise ValueError(
            "Spectrum response type must be 'transmittance' "
            "for conversion to absorbance."
        )

    if spectrum.response_unit != "%":
        raise ValueError(
            "Spectrum response unit must be '%' "
            "for conversion to absorbance."
        )

    if np.any(spectrum.response <= 0.0):
        raise ValueError(
            "Percent-transmittance values must be greater than 0 "
            "for conversion to absorbance."
        )

    converted_response = 2.0 - np.log10(
        spectrum.response
    )

    metadata = _append_transformation_provenance(
        spectrum.metadata,
        {
            "operation": (
                "irspecflow.transformations."
                "transmittance_to_absorbance"
            ),
        },
    )

    return Spectrum(
        spectrum.axis,
        converted_response,
        axis_unit=spectrum.axis_unit,
        response_type="absorbance",
        response_unit=None,
        spectrum_id=spectrum.spectrum_id,
        hierarchy=spectrum.hierarchy,
        metadata=metadata,
    )


def interpolate_spectrum(
    spectrum: Spectrum,
    target_axis: ArrayLike,
) -> Spectrum:
    """Linearly interpolate one Spectrum onto an explicit target axis.

    Parameters
    ----------
    spectrum : Spectrum
        Spectrum to interpolate. Subclasses of ``Spectrum`` are also accepted.
        The source must contain at least two spectral points, finite axis and
        response values, no duplicate finite axis values, and an axis that is
        consistently increasing or decreasing.
    target_axis : array_like
        One-dimensional real numeric values defining the requested output axis.
        The target axis must contain at least one finite value and no duplicates.
        A target containing two or more points must be strictly increasing or
        strictly decreasing. All target values must lie between the minimum and
        maximum source-axis values, including both endpoints.

    Returns
    -------
    Spectrum
        New Spectrum containing the requested target axis and the corresponding
        linearly interpolated response values. The returned object is a
        ``Spectrum`` even when the input is a ``Spectrum`` subclass. The
        target-axis order is preserved. The axis unit, response type, response
        unit, spectrum identifier, hierarchy, and unrelated metadata are
        preserved.

    Raises
    ------
    TypeError
        If ``spectrum`` is not a ``Spectrum`` instance or ``target_axis`` cannot
        be represented as real numeric array-like input.
    ValueError
        If the source spectrum does not satisfy the interpolation requirements,
        ``target_axis`` is not one-dimensional, is empty, contains non-finite or
        duplicate values, a target with two or more points is not strictly
        increasing or strictly decreasing, any target value lies outside the
        source range, or the existing provenance metadata has an incompatible
        structure.

    Notes
    -----
    Increasing and decreasing source axes are both supported, as are increasing
    and decreasing target axes. A one-point target is valid when it lies within
    the source range, and exact source-range endpoints are accepted.

    ``target_axis`` is assumed to use the same physical coordinate system and
    unit as the source axis. This function does not perform axis-unit conversion.
    It also does not extrapolate, clip out-of-range values, sort or deduplicate
    the requested target axis, or otherwise repair the source or target data.

    The supplied spectrum is not modified. Existing metadata and provenance are
    preserved, and one transformation record with ``method="linear"`` is appended
    to ``metadata["provenance"]["transformations"]``. Interpolating onto an axis
    numerically identical to the source axis still returns a distinct Spectrum
    and records the transformation.

    The source axis unit, response type, and response unit are preserved but are
    not used to determine whether the stored numerical values can be interpolated.
    Interpolation does not perform quality control or spectral preprocessing.
    """

    findings = validate_spectrum(spectrum)
    blocking_findings = _blocking_findings(
        findings,
        _INTERPOLATION_BLOCKING_CODES,
    )

    if blocking_findings:
        details = "; ".join(
            f"{finding.message} ({finding.code})"
            for finding in blocking_findings
        )

        raise ValueError(
            "Structural validation reported conditions that prevent "
            "interpolation: "
            f"{details}"
        )

    if spectrum.axis.size < 2:
        raise ValueError(
            "At least two spectral points are required for interpolation."
        )

    prepared_target_axis = _prepare_target_axis(
        target_axis
    )

    source_axis = spectrum.axis
    source_response = spectrum.response

    source_min = min(
        source_axis[0],
        source_axis[-1],
    )
    source_max = max(
        source_axis[0],
        source_axis[-1],
    )

    if (
        np.any(prepared_target_axis < source_min)
        or np.any(prepared_target_axis > source_max)
    ):
        raise ValueError(
            "target_axis values must lie within the closed "
            "numerical range of the source axis."
        )

    if source_axis[0] < source_axis[-1]:
        interpolation_axis = source_axis
        interpolation_response = source_response
    else:
        interpolation_axis = source_axis[::-1]
        interpolation_response = source_response[::-1]

    interpolated_response = np.interp(
        prepared_target_axis,
        interpolation_axis,
        interpolation_response,
    )

    metadata = _append_transformation_provenance(
        spectrum.metadata,
        {
            "operation": (
                "irspecflow.transformations."
                "interpolate_spectrum"
            ),
            "method": "linear",
        },
    )

    return Spectrum(
        prepared_target_axis,
        interpolated_response,
        axis_unit=spectrum.axis_unit,
        response_type=spectrum.response_type,
        response_unit=spectrum.response_unit,
        spectrum_id=spectrum.spectrum_id,
        hierarchy=spectrum.hierarchy,
        metadata=metadata,
    )


def _blocking_findings(
    findings: tuple[ValidationFinding, ...],
    blocking_codes: set[str],
) -> list[ValidationFinding]:
    return [
        finding
        for finding in findings
        if finding.code in blocking_codes
    ]


def _prepare_target_axis(
    target_axis: ArrayLike,
) -> NDArray[np.float64]:
    try:
        raw = np.asarray(target_axis)
    except (
        TypeError,
        ValueError,
        OverflowError,
    ) as exc:
        raise TypeError(
            "target_axis must be a one-dimensional "
            "real numeric array-like object."
        ) from exc

    if np.iscomplexobj(raw):
        raise TypeError(
            "target_axis must contain real numeric values."
        )

    try:
        prepared = np.array(
            target_axis,
            dtype=np.float64,
            copy=True,
        )
    except (
        TypeError,
        ValueError,
        OverflowError,
    ) as exc:
        raise TypeError(
            "target_axis must contain real numeric values."
        ) from exc

    if prepared.ndim != 1:
        raise ValueError(
            "target_axis must be one-dimensional."
        )

    if prepared.size == 0:
        raise ValueError(
            "target_axis must contain at least one value."
        )

    if not np.isfinite(prepared).all():
        raise ValueError(
            "target_axis must contain only finite values."
        )

    if np.unique(prepared).size != prepared.size:
        raise ValueError(
            "target_axis must not contain duplicate values."
        )

    if prepared.size > 1:
        differences = np.diff(prepared)

        if not (
            (differences > 0.0).all()
            or (differences < 0.0).all()
        ):
            raise ValueError(
                "target_axis must be strictly increasing "
                "or strictly decreasing."
            )

    return prepared


def _append_transformation_provenance(
    metadata: Mapping[str, Any],
    record: dict[str, Any],
) -> dict[str, Any]:
    prepared_metadata = deepcopy(
        dict(metadata)
    )

    if "provenance" not in prepared_metadata:
        prepared_metadata["provenance"] = {
            "transformations": [
                deepcopy(record)
            ],
        }

        return prepared_metadata

    provenance = prepared_metadata["provenance"]

    if not isinstance(provenance, Mapping):
        raise ValueError(
            "metadata['provenance'] must be a mapping."
        )

    prepared_provenance = deepcopy(
        dict(provenance)
    )

    if "transformations" not in prepared_provenance:
        prepared_provenance["transformations"] = [
            deepcopy(record)
        ]
    else:
        transformations = prepared_provenance[
            "transformations"
        ]

        if not isinstance(transformations, list):
            raise ValueError(
                "metadata['provenance']['transformations'] "
                "must be a list."
            )

        prepared_transformations = deepcopy(
            transformations
        )
        prepared_transformations.append(
            deepcopy(record)
        )
        prepared_provenance[
            "transformations"
        ] = prepared_transformations

    prepared_metadata[
        "provenance"
    ] = prepared_provenance

    return prepared_metadata