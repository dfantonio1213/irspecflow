"""Structural validation utilities for infrared spectra."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spectrum import Spectrum


__all__ = ["ValidationFinding", "validate_spectrum"]


@dataclass(frozen=True)
class ValidationFinding:
    """One structural validation finding for a ``Spectrum``.

    Parameters
    ----------
    code : str
        Stable machine-readable identifier for the detected condition.
    message : str
        Deterministic human-readable explanation of the detected condition.

    Notes
    -----
    Validation findings are immutable. The ``code`` field is the stable
    machine-readable identifier for a condition; ``message`` provides its
    human-readable explanation.
    """

    code: str
    message: str


def validate_spectrum(
    spectrum: Spectrum,
) -> tuple[ValidationFinding, ...]:
    """Inspect one Spectrum for defined structural conditions.

    Structural validation checks the stored spectral axis, response values,
    and declarations for conditions such as non-finite values, duplicate axis
    values, reversals in axis direction, and unrecognized or inconsistent
    declarations.

    Parameters
    ----------
    spectrum : Spectrum
        Spectrum to inspect for structural conditions. Subclasses of
        ``Spectrum`` are also accepted.

    Returns
    -------
    tuple of ValidationFinding
        Detected structural conditions in a consistent, predefined order.
        An empty tuple means only that no structural condition defined by this
        validator was detected.

    Raises
    ------
    TypeError
        If ``spectrum`` is not a ``Spectrum`` instance.

    Notes
    -----
    Finite spectral axes may increase or decrease. A monotonicity finding is
    produced only when the stored axis reverses direction; repeated finite
    axis values are reported separately as duplicates.

    Validation is inspection-only. It does not sort, reverse, deduplicate,
    replace non-finite values, convert units or response types, interpolate,
    perform quality control, preprocess, or otherwise modify the supplied
    spectrum.

    Declarations are compared exactly as supplied. ``"cm^-1"`` is the
    recognized axis unit, ``"absorbance"`` and ``"transmittance"`` are the
    recognized response types, and ``"%"`` is the recognized supplied response
    unit. The ``"%"`` response unit is consistent with
    ``"transmittance"`` but inconsistent with ``"absorbance"``. Missing
    ``axis_unit``, ``response_type``, and ``response_unit`` declarations do
    not produce findings. Alternate spellings, capitalization, Unicode
    notation, and surrounding whitespace are not normalized automatically.

    An empty result means that none of the structural conditions checked by
    this validator were detected. It does not establish measurement quality
    or determine whether the spectrum is suitable for a particular analysis.
    """

    if not isinstance(spectrum, Spectrum):
        raise TypeError("spectrum must be a Spectrum.")

    findings: list[ValidationFinding] = []

    axis = spectrum.axis
    response = spectrum.response

    if axis.size == 0:
        findings.append(
            ValidationFinding(
                code="empty_spectrum",
                message="Spectrum contains no spectral points.",
            )
        )
    else:
        if np.isnan(axis).any():
            findings.append(
                ValidationFinding(
                    code="axis_nan",
                    message="Axis contains one or more NaN values.",
                )
            )

        if np.isposinf(axis).any():
            findings.append(
                ValidationFinding(
                    code="axis_positive_infinity",
                    message=(
                        "Axis contains one or more positive infinity values."
                    ),
                )
            )

        if np.isneginf(axis).any():
            findings.append(
                ValidationFinding(
                    code="axis_negative_infinity",
                    message=(
                        "Axis contains one or more negative infinity values."
                    ),
                )
            )

        if np.isnan(response).any():
            findings.append(
                ValidationFinding(
                    code="response_nan",
                    message="Response contains one or more NaN values.",
                )
            )

        if np.isposinf(response).any():
            findings.append(
                ValidationFinding(
                    code="response_positive_infinity",
                    message=(
                        "Response contains one or more positive infinity values."
                    ),
                )
            )

        if np.isneginf(response).any():
            findings.append(
                ValidationFinding(
                    code="response_negative_infinity",
                    message=(
                        "Response contains one or more negative infinity values."
                    ),
                )
            )

        finite_axis = axis[np.isfinite(axis)]

        if np.unique(finite_axis).size < finite_axis.size:
            findings.append(
                ValidationFinding(
                    code="duplicate_axis_values",
                    message="Axis contains duplicate finite values.",
                )
            )

        if np.isfinite(axis).all():
            differences = np.diff(axis)
            nonzero_differences = differences[differences != 0.0]

            if (
                (nonzero_differences > 0.0).any()
                and (nonzero_differences < 0.0).any()
            ):
                findings.append(
                    ValidationFinding(
                        code="non_monotonic_axis",
                        message=(
                            "Axis reverses direction and is not consistently "
                            "increasing or decreasing."
                        ),
                    )
                )

    if (
        spectrum.axis_unit is not None
        and spectrum.axis_unit != "cm^-1"
    ):
        findings.append(
            ValidationFinding(
                code="unrecognized_axis_unit",
                message=(
                    f"Unrecognized axis unit {spectrum.axis_unit!r}; "
                    "the recognized axis unit is 'cm^-1'."
                ),
            )
        )

    if (
        spectrum.response_type is not None
        and spectrum.response_type not in {"absorbance", "transmittance"}
    ):
        findings.append(
            ValidationFinding(
                code="unrecognized_response_type",
                message=(
                    f"Unrecognized response type {spectrum.response_type!r}; "
                    "the recognized response types are 'absorbance' and "
                    "'transmittance'."
                ),
            )
        )

    if (
        spectrum.response_unit is not None
        and spectrum.response_unit != "%"
    ):
        findings.append(
            ValidationFinding(
                code="unrecognized_response_unit",
                message=(
                    f"Unrecognized response unit {spectrum.response_unit!r}; "
                    "the recognized supplied response unit is '%'."
                ),
            )
        )

    if (
        spectrum.response_type == "absorbance"
        and spectrum.response_unit == "%"
    ):
        findings.append(
            ValidationFinding(
                code="inconsistent_response_unit",
                message=(
                    "Response unit '%' is inconsistent with declared response "
                    "type 'absorbance'."
                ),
            )
        )

    return tuple(findings)