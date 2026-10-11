"""Visualization utilities for infrared spectra."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from .spectrum import Spectrum


__all__ = ["plot_spectrum", "plot_spectra"]


def plot_spectrum(
    spectrum: Spectrum,
    *,
    ax: Axes | None = None,
    label: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    axis_descending: bool | None = None,
) -> Axes:
    """Plot one Spectrum using its stored axis and response values.

    Parameters
    ----------
    spectrum : Spectrum
        Spectrum to plot. Subclasses of ``Spectrum`` are also accepted.
    ax : matplotlib.axes.Axes or None, optional
        Axes on which to draw the spectrum. When None, a new figure and axes
        are created.
    label : str or None, optional
        Explicit line label. When omitted, ``spectrum_id`` is used when
        available.
    xlabel : str or None, optional
        Explicit x-axis label. When omitted, ``"Wavenumber (cm^-1)"`` is
        used when ``axis_unit`` is exactly ``"cm^-1"``; otherwise ``"Axis"``
        is used.
    ylabel : str or None, optional
        Explicit y-axis label. When omitted, the label is derived from the
        stored response declarations. Absorbance with no response unit uses
        ``"Absorbance"``. Transmittance with ``response_unit="%"`` uses
        ``"Transmittance (%)"``, while transmittance with no response unit uses
        ``"Transmittance"``. Other or inconsistent declarations use
        ``"Response"``.
    axis_descending : bool or None, optional
        Requested x-axis display direction. None uses high-to-low display only
        when ``axis_unit`` is exactly ``"cm^-1"``. True requests high-to-low
        display and False requests low-to-high display.

    Returns
    -------
    matplotlib.axes.Axes
        Matplotlib axes containing the plotted spectrum. If ``ax`` was supplied,
        the same axes object is returned.

    Raises
    ------
    TypeError
        If a public argument has an unsupported type.

    Notes
    -----
    The stored axis and response values are plotted without sorting,
    conversion, interpolation, structural repair, or other preprocessing.
    Changing the displayed axis direction does not reorder any stored
    ``Spectrum.axis`` values.

    The supplied spectrum is not modified. This function does not create a
    legend, call ``show()``, or save a figure.
    """
    if not isinstance(spectrum, Spectrum):
        raise TypeError(
            "spectrum must be a Spectrum."
        )

    _validate_axes(ax)
    _validate_optional_string(
        label,
        "label",
    )
    _validate_optional_string(
        xlabel,
        "xlabel",
    )
    _validate_optional_string(
        ylabel,
        "ylabel",
    )
    _validate_axis_descending(
        axis_descending
    )

    if ax is None:
        _, ax = plt.subplots()

    line = ax.plot(
        spectrum.axis,
        spectrum.response,
    )[0]

    line_label = (
        spectrum.spectrum_id
        if label is None
        else label
    )

    if line_label is not None:
        line.set_label(
            line_label
        )

    ax.set_xlabel(
        _axis_label(spectrum)
        if xlabel is None
        else xlabel
    )
    ax.set_ylabel(
        _response_label(spectrum)
        if ylabel is None
        else ylabel
    )

    descending = (
        spectrum.axis_unit == "cm^-1"
        if axis_descending is None
        else axis_descending
    )

    _set_axis_direction(
        ax,
        descending,
    )

    return ax


def plot_spectra(
    spectra: Sequence[Spectrum],
    *,
    ax: Axes | None = None,
    labels: Sequence[str] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    axis_descending: bool | None = None,
) -> Axes:
    """Plot multiple Spectrum objects on the same axes.

    Parameters
    ----------
    spectra : sequence of Spectrum
        Non-empty sequence of spectra to plot. Spectrum subclasses are also
        accepted. Each spectrum is drawn from its own stored axis and response.
    ax : matplotlib.axes.Axes or None, optional
        Axes on which to draw the spectra. When None, a new figure and axes
        are created.
    labels : sequence of str or None, optional
        Explicit line labels corresponding one-to-one with ``spectra``. When
        supplied, these labels are used as provided. When omitted, each available
        ``spectrum_id`` is used.
    xlabel : str or None, optional
        Explicit x-axis label. When omitted, ``"Wavenumber (cm^-1)"`` is
        used only when every supplied spectrum declares
        ``axis_unit="cm^-1"`` exactly; otherwise ``"Axis"`` is used.
    ylabel : str or None, optional
        Explicit y-axis label. When omitted, ``"Absorbance"`` is used when
        every spectrum declares absorbance with no response unit.
        ``"Transmittance (%)"`` is used when every spectrum declares
        transmittance with ``response_unit="%"``, and ``"Transmittance"`` is
        used when every spectrum declares transmittance with no response unit.
        Other, mixed, or inconsistent declarations use ``"Response"``.
    axis_descending : bool or None, optional
        Requested x-axis display direction. None uses high-to-low display only
        when every supplied spectrum declares ``axis_unit="cm^-1"`` exactly.
        True requests high-to-low display and False requests low-to-high
        display.

    Returns
    -------
    matplotlib.axes.Axes
        Matplotlib axes containing the plotted spectra. If ``ax`` was supplied,
        the same axes object is returned.

    Raises
    ------
    TypeError
        If a public argument has an unsupported type or any supplied sequence
        element has an unsupported type.
    ValueError
        If ``spectra`` is empty or ``labels`` does not contain one entry for
        each spectrum.

    Notes
    -----
    Spectra may contain different numbers of points and different numerical
    grids. Overlaying does not perform interpolation, common-grid
    construction, sorting, conversion, repair, or structural validation.
    Changing the displayed axis direction does not reorder any stored
    ``Spectrum.axis`` values.

    The supplied spectra are not modified. This function does not create a
    legend, call ``show()``, or save a figure.
    """
    if (
        not isinstance(
            spectra,
            Sequence,
        )
        or isinstance(
            spectra,
            (str, bytes),
        )
    ):
        raise TypeError(
            "spectra must be a non-empty "
            "sequence of Spectrum objects."
        )

    prepared_spectra = tuple(
        spectra
    )

    if not prepared_spectra:
        raise ValueError(
            "spectra must contain at least one Spectrum."
        )

    for spectrum in prepared_spectra:
        if not isinstance(
            spectrum,
            Spectrum,
        ):
            raise TypeError(
                "every item in spectra must be a Spectrum."
            )

    _validate_axes(ax)
    _validate_optional_string(
        xlabel,
        "xlabel",
    )
    _validate_optional_string(
        ylabel,
        "ylabel",
    )
    _validate_axis_descending(
        axis_descending
    )

    prepared_labels: tuple[str, ...] | None = None

    if labels is not None:
        if (
            not isinstance(
                labels,
                Sequence,
            )
            or isinstance(
                labels,
                (str, bytes),
            )
        ):
            raise TypeError(
                "labels must be a sequence of strings or None."
            )

        prepared_labels = tuple(
            labels
        )

        if len(prepared_labels) != len(prepared_spectra):
            raise ValueError(
                "labels must contain one entry "
                "for each Spectrum."
            )

        for label in prepared_labels:
            if not isinstance(
                label,
                str,
            ):
                raise TypeError(
                    "every item in labels must be a string."
                )

    if ax is None:
        _, ax = plt.subplots()

    for index, spectrum in enumerate(
        prepared_spectra
    ):
        line = ax.plot(
            spectrum.axis,
            spectrum.response,
        )[0]

        line_label = (
            prepared_labels[index]
            if prepared_labels is not None
            else spectrum.spectrum_id
        )

        if line_label is not None:
            line.set_label(
                line_label
            )

    all_wavenumber = all(
        spectrum.axis_unit == "cm^-1"
        for spectrum in prepared_spectra
    )

    if xlabel is None:
        ax.set_xlabel(
            "Wavenumber (cm^-1)"
            if all_wavenumber
            else "Axis"
        )
    else:
        ax.set_xlabel(
            xlabel
        )

    if ylabel is None:
        ax.set_ylabel(
            _overlay_response_label(
                prepared_spectra
            )
        )
    else:
        ax.set_ylabel(
            ylabel
        )

    descending = (
        all_wavenumber
        if axis_descending is None
        else axis_descending
    )

    _set_axis_direction(
        ax,
        descending,
    )

    return ax


def _validate_axes(
    ax: Axes | None,
) -> None:
    if (
        ax is not None
        and not isinstance(
            ax,
            Axes,
        )
    ):
        raise TypeError(
            "ax must be a matplotlib Axes instance or None."
        )


def _validate_optional_string(
    value: str | None,
    name: str,
) -> None:
    if (
        value is not None
        and not isinstance(
            value,
            str,
        )
    ):
        raise TypeError(
            f"{name} must be a string or None."
        )


def _validate_axis_descending(
    axis_descending: bool | None,
) -> None:
    if (
        axis_descending is not None
        and not isinstance(
            axis_descending,
            bool,
        )
    ):
        raise TypeError(
            "axis_descending must be True, False, or None."
        )


def _axis_label(
    spectrum: Spectrum,
) -> str:
    if spectrum.axis_unit == "cm^-1":
        return "Wavenumber (cm^-1)"

    return "Axis"


def _response_label(
    spectrum: Spectrum,
) -> str:
    if (
        spectrum.response_type == "absorbance"
        and spectrum.response_unit is None
    ):
        return "Absorbance"

    if spectrum.response_type == "transmittance":
        if spectrum.response_unit == "%":
            return "Transmittance (%)"

        if spectrum.response_unit is None:
            return "Transmittance"

    return "Response"


def _overlay_response_label(
    spectra: Sequence[Spectrum],
) -> str:
    response_labels = [
        _response_label(spectrum)
        for spectrum in spectra
    ]

    first_label = response_labels[0]

    if first_label == "Response":
        return "Response"

    if any(
        label != first_label
        for label in response_labels[1:]
    ):
        return "Response"

    return first_label


def _set_axis_direction(
    ax: Axes,
    descending: bool,
) -> None:
    ax.xaxis.set_inverted(descending)