"""Core spectrum data structures."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["Spectrum"]


class Spectrum:
    """One-dimensional infrared spectrum with associated metadata.

    ``Spectrum`` performs only the minimal checks needed to maintain a
    consistent in-memory representation. Broader structural validation,
    including checks for missing or non-finite values, duplicated axis values,
    monotonicity, recognized units, and plausible spectrum length, belongs in
    :mod:`irspecflow.validation`.

    Parameters
    ----------
    axis : array_like
        One-dimensional spectral-axis values.
    response : array_like
        One-dimensional spectral-response values corresponding to ``axis``.
    axis_unit : str or None, optional
        Unit describing the spectral axis, if known.
    response_type : str or None, optional
        Description of the measured response, such as ``"absorbance"`` or
        ``"transmittance"``, if known.
    response_unit : str or None, optional
        Unit describing the spectral response, if applicable and known.
    spectrum_id : str or None, optional
        Identifier for the individual spectrum.
    hierarchy : Mapping[str, Any] or None, optional
        Mapping of sampling-hierarchy labels to identifiers.
    metadata : Mapping[str, Any] or None, optional
        Mapping containing acquisition, sample, instrument, provenance,
        or user-defined metadata.

    Notes
    -----
    Axis and response inputs are converted to independent ``float64`` NumPy
    arrays. The stored numerical data are exposed through read-only views,
    and the supplied ordering is preserved.

    ``hierarchy`` and ``metadata`` are deep-copied at construction so later
    changes to the caller's original mappings do not alter this object. The
    stored mappings remain editable because descriptive metadata may need to
    be completed or corrected independently of numerical processing.
    """

    def __init__(
        self,
        axis: ArrayLike,
        response: ArrayLike,
        *,
        axis_unit: str | None = None,
        response_type: str | None = None,
        response_unit: str | None = None,
        spectrum_id: str | None = None,
        hierarchy: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        axis_array = self._prepare_array(
            axis,
            "axis",
        )

        response_array = self._prepare_array(
            response,
            "response",
        )

        if axis_array.size != response_array.size:
            raise ValueError(
                "axis and response must have the same length."
            )

        self._axis = axis_array
        self._response = response_array

        self._axis_unit = self._prepare_optional_string(
            axis_unit,
            "axis_unit",
        )

        self._response_type = self._prepare_optional_string(
            response_type,
            "response_type",
        )

        self._response_unit = self._prepare_optional_string(
            response_unit,
            "response_unit",
        )

        self._spectrum_id = self._prepare_optional_string(
            spectrum_id,
            "spectrum_id",
        )

        self._hierarchy = self._prepare_mapping(
            hierarchy,
            "hierarchy",
        )

        self._metadata = self._prepare_mapping(
            metadata,
            "metadata",
        )

    @property
    def axis(self) -> NDArray[np.float64]:
        """Spectral-axis values as a read-only NumPy view."""
        return self._readonly_view(self._axis)

    @property
    def response(self) -> NDArray[np.float64]:
        """Spectral-response values as a read-only NumPy view."""
        return self._readonly_view(self._response)

    @property
    def axis_unit(self) -> str | None:
        """Declared spectral-axis unit, if known."""
        return self._axis_unit

    @property
    def response_type(self) -> str | None:
        """Declared spectral-response quantity, if known."""
        return self._response_type

    @property
    def response_unit(self) -> str | None:
        """Declared spectral-response unit, if applicable and known."""
        return self._response_unit

    @property
    def spectrum_id(self) -> str | None:
        """Optional identifier for this spectrum."""
        return self._spectrum_id

    @property
    def hierarchy(self) -> dict[str, Any]:
        """Editable sampling-hierarchy metadata copied at construction."""
        return self._hierarchy

    @property
    def metadata(self) -> dict[str, Any]:
        """Editable general metadata copied at construction."""
        return self._metadata

    @staticmethod
    def _prepare_array(
        values: ArrayLike,
        name: str,
    ) -> NDArray[np.float64]:
        try:
            raw = np.asarray(values)
        except (
            TypeError,
            ValueError,
            OverflowError,
        ) as exc:
            raise TypeError(
                f"{name} must be a one-dimensional "
                "real numeric array-like object."
            ) from exc

        if np.iscomplexobj(raw):
            raise TypeError(
                f"{name} must contain real numeric values."
            )

        try:
            array = np.array(
                values,
                dtype=np.float64,
                copy=True,
            )
        except (
            TypeError,
            ValueError,
            OverflowError,
        ) as exc:
            raise TypeError(
                f"{name} must contain real numeric values."
            ) from exc

        if array.ndim != 1:
            raise ValueError(
                f"{name} must be one-dimensional."
            )

        array.setflags(write=False)

        return array

    @staticmethod
    def _prepare_optional_string(
        value: str | None,
        name: str,
    ) -> str | None:
        if value is not None and not isinstance(value, str):
            raise TypeError(
                f"{name} must be a string or None."
            )

        return value

    @staticmethod
    def _prepare_mapping(
        value: Mapping[str, Any] | None,
        name: str,
    ) -> dict[str, Any]:
        if value is None:
            return {}

        if not isinstance(value, Mapping):
            raise TypeError(
                f"{name} must be a mapping or None."
            )

        return deepcopy(dict(value))

    @staticmethod
    def _readonly_view(
        array: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        view = array.view()
        view.setflags(write=False)

        return view