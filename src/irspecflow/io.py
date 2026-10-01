"""Spectrum import and export utilities."""

from __future__ import annotations

from codecs import lookup as lookup_codec
from collections.abc import Mapping
from copy import deepcopy
from io import StringIO
from numbers import Integral
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .spectrum import Spectrum


__all__ = ["read_spectrum"]


def read_spectrum(
    path: str | PathLike[str],
    axis_column: str | int,
    response_column: str | int,
    *,
    delimiter: str = ",",
    has_header: bool = True,
    encoding: str = "utf-8",
    comment_prefix: str | None = None,
    axis_unit: str | None = None,
    response_type: str | None = None,
    response_unit: str | None = None,
    spectrum_id: str | None = None,
    hierarchy: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Spectrum:
    """Read one local delimited-text file into a Spectrum.

    Parameters
    ----------
    path : str or path-like
        Path to one local delimited-text file.
    axis_column : str or int
        Exact column name or non-negative zero-based column position
        containing the spectral-axis values.
    response_column : str or int
        Exact column name or non-negative zero-based column position
        containing the spectral-response values.
    delimiter : str, optional
        Single literal character separating fields. Defaults to a comma.
    has_header : bool, optional
        Whether the first parsed table row contains column names.
        Defaults to True.
    encoding : str, optional
        Text encoding used to read the file. Defaults to ``"utf-8"``.
    comment_prefix : str or None, optional
        Single non-whitespace character marking leading comment or metadata
        lines. Leading lines whose first non-whitespace character matches
        this value are skipped before table parsing begins. Comment-like
        lines after the table has begun are treated as table content.
        Defaults to None.
    axis_unit : str or None, optional
        Declared unit of the spectral axis.
    response_type : str or None, optional
        Declared response quantity, such as ``"absorbance"`` or
        ``"transmittance"``.
    response_unit : str or None, optional
        Declared unit of the spectral response, if applicable.
    spectrum_id : str or None, optional
        Identifier for the imported spectrum.
    hierarchy : Mapping[str, Any] or None, optional
        Sampling-hierarchy metadata passed to the returned ``Spectrum``.
    metadata : Mapping[str, Any] or None, optional
        Caller-supplied metadata. Import provenance is added without
        modifying the caller's original mapping.

    Returns
    -------
    Spectrum
        Spectrum containing the selected numerical columns, descriptive
        fields, and import provenance.

    Raises
    ------
    TypeError
        If a public argument has an unsupported type.
    ValueError
        If an argument value is invalid, the table cannot be parsed, a
        selected value cannot be represented numerically, or import
        provenance conflicts with caller metadata.
    KeyError
        If a requested named column is unavailable.
    IndexError
        If a requested positional column is outside the parsed table.
    OSError
        If the local file cannot be accessed.
    UnicodeDecodeError
        If the file cannot be decoded using the requested encoding.

    Notes
    -----
    Only the selected axis and response columns must be numerically
    representable. Parsed row order is preserved.

    This function performs file ingestion only. It does not sort the axis,
    validate spectral structure or units, convert response or axis units,
    interpolate, perform quality control, or apply spectral preprocessing.
    """

    source_path = _prepare_path(path)
    delimiter = _prepare_delimiter(delimiter)
    has_header = _prepare_header_setting(has_header)
    encoding = _prepare_encoding(encoding)
    comment_prefix = _prepare_comment_prefix(comment_prefix)

    prepared_axis_column = _prepare_column_selector(
        axis_column,
        "axis_column",
        has_header,
    )

    prepared_response_column = _prepare_column_selector(
        response_column,
        "response_column",
        has_header,
    )

    table = _read_table(
        source_path,
        delimiter,
        encoding,
        comment_prefix,
    )

    if has_header:
        column_names = table.iloc[0].tolist()
        data = table.iloc[1:]
    else:
        column_names = None
        data = table

    axis_index = _resolve_column(
        prepared_axis_column,
        "axis_column",
        column_names,
        table.shape[1],
    )

    response_index = _resolve_column(
        prepared_response_column,
        "response_column",
        column_names,
        table.shape[1],
    )

    axis = _convert_numeric_values(
        data.iloc[:, axis_index],
        "axis",
    )

    response = _convert_numeric_values(
        data.iloc[:, response_index],
        "response",
    )

    import_record = {
        "importer": "irspecflow.io.read_spectrum",
        "source_filename": source_path.name,
        "axis_column": axis_column,
        "response_column": response_column,
        "delimiter": delimiter,
        "has_header": has_header,
        "encoding": encoding,
        "comment_prefix": comment_prefix,
    }

    prepared_metadata = _merge_import_provenance(
        metadata,
        import_record,
    )

    return Spectrum(
        axis,
        response,
        axis_unit=axis_unit,
        response_type=response_type,
        response_unit=response_unit,
        spectrum_id=spectrum_id,
        hierarchy=hierarchy,
        metadata=prepared_metadata,
    )


def _prepare_path(
    path: str | PathLike[str],
) -> Path:
    if not isinstance(path, (str, PathLike)):
        raise TypeError(
            "path must be a string or path-like object."
        )

    try:
        return Path(path)
    except TypeError as exc:
        raise TypeError(
            "path must be a string or path-like object."
        ) from exc


def _prepare_delimiter(
    delimiter: str,
) -> str:
    if not isinstance(delimiter, str):
        raise TypeError(
            "delimiter must be a string."
        )

    if len(delimiter) != 1 or delimiter in {"\r", "\n"}:
        raise ValueError(
            "delimiter must be one literal character."
        )

    return delimiter


def _prepare_header_setting(
    has_header: bool,
) -> bool:
    if not isinstance(has_header, bool):
        raise TypeError(
            "has_header must be a Boolean value."
        )

    return has_header


def _prepare_encoding(
    encoding: str,
) -> str:
    if not isinstance(encoding, str):
        raise TypeError(
            "encoding must be a string."
        )

    try:
        lookup_codec(encoding)
    except LookupError as exc:
        raise ValueError(
            f"Unknown text encoding: {encoding!r}."
        ) from exc

    return encoding


def _prepare_comment_prefix(
    comment_prefix: str | None,
) -> str | None:
    if comment_prefix is None:
        return None

    if not isinstance(comment_prefix, str):
        raise TypeError(
            "comment_prefix must be a string or None."
        )

    if (
        len(comment_prefix) != 1
        or comment_prefix.isspace()
    ):
        raise ValueError(
            "comment_prefix must be one non-whitespace "
            "literal character."
        )

    return comment_prefix


def _prepare_column_selector(
    selector: str | int,
    name: str,
    has_header: bool,
) -> str | int:
    if isinstance(selector, bool):
        raise TypeError(
            f"{name} must be a column name or "
            "non-negative integer position."
        )

    if isinstance(selector, Integral):
        position = int(selector)

        if position < 0:
            raise ValueError(
                f"{name} must not be negative."
            )

        return position

    if isinstance(selector, str):
        if not has_header:
            raise ValueError(
                f"{name} must use an integer position "
                "when has_header is False."
            )

        return selector

    raise TypeError(
        f"{name} must be a column name or "
        "non-negative integer position."
    )


def _remove_leading_comments(
    lines: list[str],
    comment_prefix: str,
) -> list[str]:
    start_index = 0

    for index, line in enumerate(lines):
        if (
            not line.strip()
            or line.lstrip().startswith(comment_prefix)
        ):
            start_index = index + 1
            continue

        break

    return lines[start_index:]


def _read_table(
    path: Path,
    delimiter: str,
    encoding: str,
    comment_prefix: str | None,
) -> pd.DataFrame:
    with path.open(
        "r",
        encoding=encoding,
        newline="",
    ) as file:
        lines = file.readlines()

    if comment_prefix is not None:
        lines = _remove_leading_comments(
            lines,
            comment_prefix,
        )

    text = "".join(lines)

    try:
        return pd.read_csv(
            StringIO(text),
            sep=delimiter,
            header=None,
            dtype=str,
            keep_default_na=False,
        )
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as exc:
        raise ValueError(
            f"Could not parse {path.name!r} "
            "as a delimited-text table."
        ) from exc


def _resolve_column(
    selector: str | int,
    name: str,
    column_names: list[str] | None,
    column_count: int,
) -> int:
    if isinstance(selector, int):
        if selector >= column_count:
            raise IndexError(
                f"{name} position {selector} is outside "
                "the parsed table."
            )

        return selector

    assert column_names is not None

    matches = [
        index
        for index, column_name in enumerate(column_names)
        if column_name == selector
    ]

    if not matches:
        raise KeyError(
            f"{name} {selector!r} was not found."
        )

    if len(matches) > 1:
        raise ValueError(
            f"{name} {selector!r} is ambiguous; "
            "use an integer position instead."
        )

    return matches[0]


def _convert_numeric_values(
    values: pd.Series,
    name: str,
) -> NDArray[np.float64]:
    converted = np.empty(
        len(values),
        dtype=np.float64,
    )

    for index, value in enumerate(values):
        if value == "":
            converted[index] = np.nan
            continue

        try:
            converted[index] = float(value)
        except (
            TypeError,
            ValueError,
            OverflowError,
        ) as exc:
            raise ValueError(
                f"Selected {name} column contains "
                f"a nonnumeric value: {value!r}."
            ) from exc

    return converted


def _merge_import_provenance(
    metadata: Mapping[str, Any] | None,
    import_record: dict[str, Any],
) -> dict[str, Any]:
    if metadata is None:
        prepared_metadata = {}
    elif not isinstance(metadata, Mapping):
        raise TypeError(
            "metadata must be a mapping or None."
        )
    else:
        prepared_metadata = deepcopy(
            dict(metadata)
        )

    if "provenance" not in prepared_metadata:
        prepared_metadata["provenance"] = {
            "import": import_record,
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

    if "import" in prepared_provenance:
        if prepared_provenance["import"] != import_record:
            raise ValueError(
                "metadata['provenance']['import'] "
                "conflicts with the generated import provenance."
            )
    else:
        prepared_provenance["import"] = import_record

    prepared_metadata["provenance"] = prepared_provenance

    return prepared_metadata