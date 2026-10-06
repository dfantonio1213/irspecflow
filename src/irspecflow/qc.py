"""Quality-control utilities for infrared spectra."""

from __future__ import annotations

from .spectrum import Spectrum
from .validation import validate_spectrum


__all__ = ["response_range"]


_RESPONSE_RANGE_BLOCKING_CODES = {
    "empty_spectrum",
    "response_nan",
    "response_positive_infinity",
    "response_negative_infinity",
}


def response_range(
    spectrum: Spectrum,
) -> float:
    """Return the numerical span of one Spectrum response.

    Parameters
    ----------
    spectrum : Spectrum
        Spectrum whose stored response values are inspected. Subclasses of
        ``Spectrum`` are also accepted.

    Returns
    -------
    float
        Maximum stored response value minus the minimum stored response value.
        A one-point finite spectrum returns 0.0.

    Raises
    ------
    TypeError
        If ``spectrum`` is not a ``Spectrum`` instance.
    ValueError
        If the spectrum contains no spectral points or its response contains
        NaN, positive infinity, or negative infinity values.

    Notes
    -----
    Response range is representation-dependent. Values calculated from
    different response representations, such as absorbance and percent
    transmittance, are not intrinsically comparable.

    The calculation uses only the stored response values. Conditions limited to
    the spectral axis, or missing, unrecognized, or inconsistent axis or response
    declarations, do not prevent the calculation.

    This function does not modify the supplied spectrum or store the calculated
    range in metadata. It does not apply a threshold or classify the spectrum as
    passing or failing quality control.
    """
    findings = validate_spectrum(
        spectrum
    )

    blocking_findings = [
        finding
        for finding in findings
        if finding.code
        in _RESPONSE_RANGE_BLOCKING_CODES
    ]

    if blocking_findings:
        details = "; ".join(
            f"{finding.message} ({finding.code})"
            for finding in blocking_findings
        )

        raise ValueError(
            "Structural validation reported conditions "
            "that prevent response-range calculation: "
            f"{details}"
        )

    response = spectrum.response

    return float(
        response.max()
        - response.min()
    )