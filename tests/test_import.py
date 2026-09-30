from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

import irspecflow
from irspecflow import Spectrum
from irspecflow.io import read_spectrum


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_package_import():
    import irspecflow


def test_read_spectrum_is_available_through_io():
    import irspecflow.io

    assert irspecflow.io.read_spectrum is read_spectrum


def test_read_spectrum_is_not_exposed_at_package_level():
    assert not hasattr(irspecflow, "read_spectrum")


def test_read_spectrum_returns_spectrum():
    spectrum = read_spectrum(
        DATA_DIR / "example_spectrum.csv",
        axis_column="wavenumber",
        response_column="absorbance",
    )

    assert isinstance(spectrum, Spectrum)


def test_read_spectrum_accepts_path_string():
    spectrum = read_spectrum(
        str(DATA_DIR / "example_spectrum.csv"),
        axis_column="wavenumber",
        response_column="absorbance",
    )

    np.testing.assert_array_equal(
        spectrum.axis,
        [4000.0, 3000.0, 2000.0, 1000.0, 400.0],
    )


def test_read_spectrum_accepts_path_like_object():
    spectrum = read_spectrum(
        DATA_DIR / "example_spectrum.csv",
        axis_column="wavenumber",
        response_column="absorbance",
    )

    np.testing.assert_array_equal(
        spectrum.response,
        [0.080, 0.120, 0.200, 0.350, 0.180],
    )


@pytest.mark.parametrize(
    "path",
    [
        None,
        1.5,
        [],
        {"path": "spectrum.csv"},
    ],
)
def test_read_spectrum_rejects_invalid_path_type(
    path,
):
    with pytest.raises(
        TypeError,
        match="path",
    ):
        read_spectrum(
            path,
            axis_column="wavenumber",
            response_column="absorbance",
        )


def test_read_spectrum_reads_named_columns_with_header():
    spectrum = read_spectrum(
        DATA_DIR / "example_spectrum.csv",
        axis_column="wavenumber",
        response_column="absorbance",
    )

    np.testing.assert_array_equal(
        spectrum.axis,
        [4000.0, 3000.0, 2000.0, 1000.0, 400.0],
    )

    np.testing.assert_array_equal(
        spectrum.response,
        [0.080, 0.120, 0.200, 0.350, 0.180],
    )


def test_read_spectrum_reads_positional_columns_with_header():
    spectrum = read_spectrum(
        DATA_DIR / "example_spectrum.csv",
        axis_column=0,
        response_column=1,
    )

    np.testing.assert_array_equal(
        spectrum.axis,
        [4000.0, 3000.0, 2000.0, 1000.0, 400.0],
    )

    np.testing.assert_array_equal(
        spectrum.response,
        [0.080, 0.120, 0.200, 0.350, 0.180],
    )


def test_read_spectrum_reads_headerless_tab_delimited_input():
    spectrum = read_spectrum(
        DATA_DIR / "example_spectrum_headerless.tsv",
        axis_column=0,
        response_column=1,
        delimiter="\t",
        has_header=False,
    )

    np.testing.assert_array_equal(
        spectrum.axis,
        [4000.0, 3000.0, 2000.0, 1000.0, 400.0],
    )

    np.testing.assert_array_equal(
        spectrum.response,
        [0.080, 0.120, 0.200, 0.350, 0.180],
    )


def test_read_spectrum_accepts_another_literal_delimiter(
    tmp_path,
):
    path = tmp_path / "spectrum.txt"

    path.write_text(
        "axis;response\n"
        "4000;0.10\n"
        "3000;0.20\n",
        encoding="utf-8",
    )

    spectrum = read_spectrum(
        path,
        axis_column="axis",
        response_column="response",
        delimiter=";",
    )

    np.testing.assert_array_equal(
        spectrum.axis,
        [4000.0, 3000.0],
    )

    np.testing.assert_array_equal(
        spectrum.response,
        [0.10, 0.20],
    )


def test_read_spectrum_passes_descriptive_fields():
    spectrum = read_spectrum(
        DATA_DIR / "example_spectrum.csv",
        axis_column="wavenumber",
        response_column="absorbance",
        axis_unit="cm^-1",
        response_type="absorbance",
        response_unit="AU",
        spectrum_id="synthetic-spectrum",
        hierarchy={
            "sample": "sample-a",
            "replicate": 1,
        },
    )

    assert spectrum.axis_unit == "cm^-1"
    assert spectrum.response_type == "absorbance"
    assert spectrum.response_unit == "AU"
    assert spectrum.spectrum_id == "synthetic-spectrum"

    assert spectrum.hierarchy == {
        "sample": "sample-a",
        "replicate": 1,
    }


def test_read_spectrum_preserves_caller_metadata():
    spectrum = read_spectrum(
        DATA_DIR / "example_spectrum.csv",
        axis_column="wavenumber",
        response_column="absorbance",
        metadata={
            "sample": {
                "source": "synthetic",
            }
        },
    )

    assert spectrum.metadata["sample"] == {
        "source": "synthetic",
    }


def test_read_spectrum_ignores_unused_text_columns():
    spectrum = read_spectrum(
        DATA_DIR / "example_spectrum.csv",
        axis_column="wavenumber",
        response_column="absorbance",
    )

    np.testing.assert_array_equal(
        spectrum.axis,
        [4000.0, 3000.0, 2000.0, 1000.0, 400.0],
    )


def test_read_spectrum_allows_same_source_column_for_axis_and_response():
    spectrum = read_spectrum(
        DATA_DIR / "example_spectrum.csv",
        axis_column="wavenumber",
        response_column="wavenumber",
    )

    np.testing.assert_array_equal(
        spectrum.axis,
        spectrum.response,
    )


@pytest.mark.parametrize(
    "axis",
    [
        [1000.0, 2000.0, 3000.0],
        [3000.0, 2000.0, 1000.0],
        [3000.0, 1000.0, 2000.0],
        [3000.0, 2000.0, 2000.0],
    ],
)
def test_read_spectrum_preserves_axis_values_and_order(
    tmp_path,
    axis,
):
    path = tmp_path / "spectrum.csv"

    path.write_text(
        "axis,response\n"
        f"{axis[0]},0.10\n"
        f"{axis[1]},0.20\n"
        f"{axis[2]},0.30\n",
        encoding="utf-8",
    )

    spectrum = read_spectrum(
        path,
        axis_column="axis",
        response_column="response",
    )

    np.testing.assert_array_equal(
        spectrum.axis,
        axis,
    )


def test_read_spectrum_preserves_ordinary_floating_point_values(
    tmp_path,
):
    path = tmp_path / "spectrum.csv"

    path.write_text(
        "axis,response\n"
        "4000.25,0.12345\n"
        "3000.75,-0.98765\n",
        encoding="utf-8",
    )

    spectrum = read_spectrum(
        path,
        axis_column="axis",
        response_column="response",
    )

    np.testing.assert_array_equal(
        spectrum.axis,
        [4000.25, 3000.75],
    )

    np.testing.assert_array_equal(
        spectrum.response,
        [0.12345, -0.98765],
    )


def test_read_spectrum_accepts_scientific_notation(
    tmp_path,
):
    path = tmp_path / "spectrum.csv"

    path.write_text(
        "axis,response\n"
        "4.0e3,8.0e-2\n"
        "3.0e3,-1.2e-1\n",
        encoding="utf-8",
    )

    spectrum = read_spectrum(
        path,
        axis_column="axis",
        response_column="response",
    )

    np.testing.assert_array_equal(
        spectrum.axis,
        [4000.0, 3000.0],
    )

    np.testing.assert_array_equal(
        spectrum.response,
        [0.08, -0.12],
    )


def test_read_spectrum_preserves_nonfinite_values(
    tmp_path,
):
    path = tmp_path / "spectrum.csv"

    path.write_text(
        "axis,response\n"
        "NaN,inf\n"
        "inf,-inf\n"
        "-inf,NaN\n",
        encoding="utf-8",
    )

    spectrum = read_spectrum(
        path,
        axis_column="axis",
        response_column="response",
    )

    assert np.isnan(spectrum.axis[0])
    assert np.isposinf(spectrum.axis[1])
    assert np.isneginf(spectrum.axis[2])

    assert np.isposinf(spectrum.response[0])
    assert np.isneginf(spectrum.response[1])
    assert np.isnan(spectrum.response[2])


def test_read_spectrum_preserves_empty_selected_field_as_nan(
    tmp_path,
):
    path = tmp_path / "spectrum.csv"

    path.write_text(
        "axis,response\n"
        "4000,0.10\n"
        ",0.20\n"
        "2000,\n",
        encoding="utf-8",
    )

    spectrum = read_spectrum(
        path,
        axis_column="axis",
        response_column="response",
    )

    assert np.isnan(spectrum.axis[1])
    assert np.isnan(spectrum.response[2])

    assert spectrum.axis[0] == 4000.0
    assert spectrum.axis[2] == 2000.0
    assert spectrum.response[0] == 0.10
    assert spectrum.response[1] == 0.20


def test_read_spectrum_allows_valid_zero_row_spectrum(
    tmp_path,
):
    path = tmp_path / "spectrum.csv"

    path.write_text(
        "axis,response\n",
        encoding="utf-8",
    )

    spectrum = read_spectrum(
        path,
        axis_column="axis",
        response_column="response",
    )

    assert spectrum.axis.size == 0
    assert spectrum.response.size == 0


def test_read_spectrum_accepts_explicit_encoding(
    tmp_path,
):
    path = tmp_path / "latin1.csv"

    path.write_text(
        "axis,response,note\n"
        "4000,0.10,café\n",
        encoding="latin-1",
    )

    spectrum = read_spectrum(
        path,
        axis_column="axis",
        response_column="response",
        encoding="latin-1",
    )

    np.testing.assert_array_equal(
        spectrum.axis,
        [4000.0],
    )

    np.testing.assert_array_equal(
        spectrum.response,
        [0.10],
    )

    assert (
        spectrum.metadata["provenance"]["import"]["encoding"]
        == "latin-1"
    )


def test_read_spectrum_skips_leading_comment_lines(
    tmp_path,
):
    path = tmp_path / "instrument_export.csv"

    path.write_text(
        "# Instrument: Example FTIR\n"
        "# Resolution: 4 cm-1\n"
        "# Exported spectrum\n"
        "wavenumber,absorbance\n"
        "4000,0.08\n"
        "3000,0.12\n",
        encoding="utf-8",
    )

    spectrum = read_spectrum(
        path,
        axis_column="wavenumber",
        response_column="absorbance",
        comment_prefix="#",
    )

    np.testing.assert_array_equal(
        spectrum.axis,
        [4000.0, 3000.0],
    )

    np.testing.assert_array_equal(
        spectrum.response,
        [0.08, 0.12],
    )


def test_read_spectrum_accepts_another_comment_prefix(
    tmp_path,
):
    path = tmp_path / "instrument_export.csv"

    path.write_text(
        "; Instrument metadata\n"
        "; Exported spectrum\n"
        "axis,response\n"
        "4000,0.10\n"
        "3000,0.20\n",
        encoding="utf-8",
    )

    spectrum = read_spectrum(
        path,
        axis_column="axis",
        response_column="response",
        comment_prefix=";",
    )

    np.testing.assert_array_equal(
        spectrum.axis,
        [4000.0, 3000.0],
    )

    np.testing.assert_array_equal(
        spectrum.response,
        [0.10, 0.20],
    )


def test_read_spectrum_rejects_comment_only_file(
    tmp_path,
):
    path = tmp_path / "comments_only.csv"

    path.write_text(
        "# Instrument: Example FTIR\n"
        "# Resolution: 4 cm-1\n"
        "# No spectral table follows\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Could not parse",
    ):
        read_spectrum(
            path,
            axis_column="axis",
            response_column="response",
            comment_prefix="#",
        )


def test_read_spectrum_does_not_skip_comments_by_default(
    tmp_path,
):
    path = tmp_path / "instrument_export.csv"

    path.write_text(
        "# Instrument: Example FTIR\n"
        "wavenumber,absorbance\n"
        "4000,0.08\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Could not parse",
    ):
        read_spectrum(
            path,
            axis_column="wavenumber",
            response_column="absorbance",
        )


def test_read_spectrum_skips_indented_leading_comments(
    tmp_path,
):
    path = tmp_path / "instrument_export.csv"

    path.write_text(
        "   # Instrument metadata\n"
        "\t# Another comment\n"
        "axis,response\n"
        "4000,0.10\n",
        encoding="utf-8",
    )

    spectrum = read_spectrum(
        path,
        axis_column="axis",
        response_column="response",
        comment_prefix="#",
    )

    np.testing.assert_array_equal(
        spectrum.axis,
        [4000.0],
    )


def test_read_spectrum_skips_blank_lines_with_leading_comments(
    tmp_path,
):
    path = tmp_path / "instrument_export.csv"

    path.write_text(
        "\n"
        "# Instrument: Example FTIR\n"
        "\n"
        "   # Resolution: 4 cm-1\n"
        "\n"
        "axis,response\n"
        "4000,0.10\n"
        "3000,0.20\n",
        encoding="utf-8",
    )

    spectrum = read_spectrum(
        path,
        axis_column="axis",
        response_column="response",
        comment_prefix="#",
    )

    np.testing.assert_array_equal(
        spectrum.axis,
        [4000.0, 3000.0],
    )

    np.testing.assert_array_equal(
        spectrum.response,
        [0.10, 0.20],
    )


def test_read_spectrum_skips_comments_before_headerless_data(
    tmp_path,
):
    path = tmp_path / "instrument_export.tsv"

    path.write_text(
        "# Instrument metadata\n"
        "# Data follows\n"
        "4000\t0.08\n"
        "3000\t0.12\n",
        encoding="utf-8",
    )

    spectrum = read_spectrum(
        path,
        axis_column=0,
        response_column=1,
        delimiter="\t",
        has_header=False,
        comment_prefix="#",
    )

    np.testing.assert_array_equal(
        spectrum.axis,
        [4000.0, 3000.0],
    )

    np.testing.assert_array_equal(
        spectrum.response,
        [0.08, 0.12],
    )


def test_read_spectrum_does_not_treat_comments_after_table_start_as_metadata(
    tmp_path,
):
    path = tmp_path / "spectrum.csv"

    path.write_text(
        "# Instrument metadata\n"
        "axis,response\n"
        "4000,0.10\n"
        "# unexpected line\n"
        "3000,0.20\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="axis",
    ):
        read_spectrum(
            path,
            axis_column="axis",
            response_column="response",
            comment_prefix="#",
        )


def test_read_spectrum_records_import_provenance():
    path = DATA_DIR / "example_spectrum.csv"

    spectrum = read_spectrum(
        path,
        axis_column="wavenumber",
        response_column="absorbance",
    )

    assert spectrum.metadata["provenance"]["import"] == {
        "importer": "irspecflow.io.read_spectrum",
        "source_filename": "example_spectrum.csv",
        "axis_column": "wavenumber",
        "response_column": "absorbance",
        "delimiter": ",",
        "has_header": True,
        "encoding": "utf-8",
        "comment_prefix": None,
    }


def test_read_spectrum_provenance_does_not_store_absolute_path():
    path = DATA_DIR / "example_spectrum.csv"

    spectrum = read_spectrum(
        path,
        axis_column="wavenumber",
        response_column="absorbance",
    )

    import_record = spectrum.metadata["provenance"]["import"]

    assert import_record["source_filename"] == path.name
    assert str(path.resolve()) not in repr(import_record)


def test_read_spectrum_provenance_preserves_original_selectors(
    tmp_path,
):
    path = tmp_path / "spectrum.csv"

    path.write_text(
        "axis,response\n"
        "4000,0.10\n",
        encoding="utf-8",
    )

    spectrum = read_spectrum(
        path,
        axis_column=0,
        response_column=1,
    )

    import_record = spectrum.metadata["provenance"]["import"]

    assert import_record["axis_column"] == 0
    assert import_record["response_column"] == 1


def test_read_spectrum_records_nondefault_parsing_settings(
    tmp_path,
):
    path = tmp_path / "instrument_export.txt"

    path.write_text(
        "# Metadata\n"
        "4000;0.10\n",
        encoding="utf-8",
    )

    spectrum = read_spectrum(
        path,
        axis_column=0,
        response_column=1,
        delimiter=";",
        has_header=False,
        comment_prefix="#",
    )

    import_record = spectrum.metadata["provenance"]["import"]

    assert import_record["delimiter"] == ";"
    assert import_record["has_header"] is False
    assert import_record["encoding"] == "utf-8"
    assert import_record["comment_prefix"] == "#"


def test_read_spectrum_preserves_unrelated_provenance():
    spectrum = read_spectrum(
        DATA_DIR / "example_spectrum.csv",
        axis_column="wavenumber",
        response_column="absorbance",
        metadata={
            "provenance": {
                "operator": "analyst-a",
            }
        },
    )

    assert (
        spectrum.metadata["provenance"]["operator"]
        == "analyst-a"
    )

    assert (
        "import"
        in spectrum.metadata["provenance"]
    )


def test_read_spectrum_does_not_mutate_caller_metadata():
    metadata = {
        "sample": {
            "source": "synthetic",
        },
        "provenance": {
            "operator": "analyst-a",
        },
    }

    original_metadata = deepcopy(metadata)

    read_spectrum(
        DATA_DIR / "example_spectrum.csv",
        axis_column="wavenumber",
        response_column="absorbance",
        metadata=metadata,
    )

    assert metadata == original_metadata


def test_read_spectrum_accepts_identical_existing_import_provenance():
    import_record = {
        "importer": "irspecflow.io.read_spectrum",
        "source_filename": "example_spectrum.csv",
        "axis_column": "wavenumber",
        "response_column": "absorbance",
        "delimiter": ",",
        "has_header": True,
        "encoding": "utf-8",
        "comment_prefix": None,
    }

    spectrum = read_spectrum(
        DATA_DIR / "example_spectrum.csv",
        axis_column="wavenumber",
        response_column="absorbance",
        metadata={
            "provenance": {
                "import": import_record,
            }
        },
    )

    assert (
        spectrum.metadata["provenance"]["import"]
        == import_record
    )


def test_read_spectrum_rejects_conflicting_existing_import_provenance():
    conflicting_record = {
        "importer": "irspecflow.io.read_spectrum",
        "source_filename": "example_spectrum.csv",
        "axis_column": "wavenumber",
        "response_column": "absorbance",
        "delimiter": "\t",
        "has_header": True,
        "encoding": "utf-8",
        "comment_prefix": None,
    }

    with pytest.raises(
        ValueError,
        match="conflicts",
    ):
        read_spectrum(
            DATA_DIR / "example_spectrum.csv",
            axis_column="wavenumber",
            response_column="absorbance",
            metadata={
                "provenance": {
                    "import": conflicting_record,
                }
            },
        )


def test_read_spectrum_rejects_nonmapping_provenance():
    with pytest.raises(
        ValueError,
        match="provenance",
    ):
        read_spectrum(
            DATA_DIR / "example_spectrum.csv",
            axis_column="wavenumber",
            response_column="absorbance",
            metadata={
                "provenance": "manual",
            },
        )


def test_read_spectrum_preserves_missing_file_error(
    tmp_path,
):
    path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        read_spectrum(
            path,
            axis_column="axis",
            response_column="response",
        )


def test_read_spectrum_preserves_decoding_error(
    tmp_path,
):
    path = tmp_path / "latin1.csv"

    path.write_text(
        "axis,response,note\n"
        "4000,0.10,café\n",
        encoding="latin-1",
    )

    with pytest.raises(UnicodeDecodeError):
        read_spectrum(
            path,
            axis_column="axis",
            response_column="response",
        )


def test_read_spectrum_rejects_malformed_table(
    tmp_path,
):
    path = tmp_path / "malformed.csv"

    path.write_text(
        "axis,response\n"
        "4000,0.10\n"
        "3000,0.20,unexpected\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Could not parse",
    ):
        read_spectrum(
            path,
            axis_column="axis",
            response_column="response",
        )


def test_read_spectrum_rejects_empty_file(
    tmp_path,
):
    path = tmp_path / "empty.csv"

    path.write_text(
        "",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Could not parse",
    ):
        read_spectrum(
            path,
            axis_column="axis",
            response_column="response",
        )


def test_read_spectrum_rejects_unavailable_named_column():
    with pytest.raises(
        KeyError,
        match="missing",
    ):
        read_spectrum(
            DATA_DIR / "example_spectrum.csv",
            axis_column="missing",
            response_column="absorbance",
        )


def test_read_spectrum_named_columns_are_case_sensitive():
    with pytest.raises(
        KeyError,
        match="Wavenumber",
    ):
        read_spectrum(
            DATA_DIR / "example_spectrum.csv",
            axis_column="Wavenumber",
            response_column="absorbance",
        )


@pytest.mark.parametrize(
    ("axis_column", "response_column"),
    [
        (3, 1),
        (0, 3),
    ],
)
def test_read_spectrum_rejects_out_of_range_position(
    axis_column,
    response_column,
):
    with pytest.raises(IndexError):
        read_spectrum(
            DATA_DIR / "example_spectrum.csv",
            axis_column=axis_column,
            response_column=response_column,
        )


@pytest.mark.parametrize(
    ("axis_column", "response_column"),
    [
        (-1, 1),
        (0, -1),
    ],
)
def test_read_spectrum_rejects_negative_position(
    axis_column,
    response_column,
):
    with pytest.raises(
        ValueError,
        match="must not be negative",
    ):
        read_spectrum(
            DATA_DIR / "example_spectrum.csv",
            axis_column=axis_column,
            response_column=response_column,
        )


@pytest.mark.parametrize(
    ("axis_column", "response_column"),
    [
        (True, 1),
        (0, False),
    ],
)
def test_read_spectrum_rejects_boolean_selector(
    axis_column,
    response_column,
):
    with pytest.raises(
        TypeError,
        match="column name or non-negative integer",
    ):
        read_spectrum(
            DATA_DIR / "example_spectrum.csv",
            axis_column=axis_column,
            response_column=response_column,
        )


@pytest.mark.parametrize(
    ("axis_column", "response_column"),
    [
        (1.5, 1),
        (0, 1.5),
        (["wavenumber"], 1),
        (0, {"response": 1}),
    ],
)
def test_read_spectrum_rejects_invalid_selector_type(
    axis_column,
    response_column,
):
    with pytest.raises(
        TypeError,
        match="column name or non-negative integer",
    ):
        read_spectrum(
            DATA_DIR / "example_spectrum.csv",
            axis_column=axis_column,
            response_column=response_column,
        )


@pytest.mark.parametrize(
    ("axis_column", "response_column"),
    [
        ("axis", 1),
        (0, "response"),
    ],
)
def test_read_spectrum_rejects_string_selector_without_header(
    axis_column,
    response_column,
):
    with pytest.raises(
        ValueError,
        match="has_header is False",
    ):
        read_spectrum(
            DATA_DIR / "example_spectrum_headerless.tsv",
            axis_column=axis_column,
            response_column=response_column,
            delimiter="\t",
            has_header=False,
        )


@pytest.mark.parametrize(
    "value",
    [
        "N/A",
        "NULL",
        "missing",
        "not-numeric",
    ],
)
def test_read_spectrum_rejects_nonnumeric_axis_value(
    tmp_path,
    value,
):
    path = tmp_path / "spectrum.csv"

    path.write_text(
        "axis,response\n"
        f"{value},0.10\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="axis",
    ):
        read_spectrum(
            path,
            axis_column="axis",
            response_column="response",
        )


@pytest.mark.parametrize(
    "value",
    [
        "N/A",
        "NULL",
        "missing",
        "not-numeric",
    ],
)
def test_read_spectrum_rejects_nonnumeric_response_value(
    tmp_path,
    value,
):
    path = tmp_path / "spectrum.csv"

    path.write_text(
        "axis,response\n"
        f"4000,{value}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="response",
    ):
        read_spectrum(
            path,
            axis_column="axis",
            response_column="response",
        )


@pytest.mark.parametrize(
    "delimiter",
    [
        "",
        "::",
        "\n",
        "\r",
    ],
)
def test_read_spectrum_rejects_invalid_delimiter_value(
    delimiter,
):
    with pytest.raises(
        ValueError,
        match="delimiter",
    ):
        read_spectrum(
            DATA_DIR / "example_spectrum.csv",
            axis_column="wavenumber",
            response_column="absorbance",
            delimiter=delimiter,
        )


@pytest.mark.parametrize(
    "delimiter",
    [
        1,
        True,
        [";"],
    ],
)
def test_read_spectrum_rejects_invalid_delimiter_type(
    delimiter,
):
    with pytest.raises(
        TypeError,
        match="delimiter",
    ):
        read_spectrum(
            DATA_DIR / "example_spectrum.csv",
            axis_column="wavenumber",
            response_column="absorbance",
            delimiter=delimiter,
        )


@pytest.mark.parametrize(
    "has_header",
    [
        1,
        0,
        "yes",
        None,
    ],
)
def test_read_spectrum_rejects_invalid_header_setting(
    has_header,
):
    with pytest.raises(
        TypeError,
        match="has_header",
    ):
        read_spectrum(
            DATA_DIR / "example_spectrum.csv",
            axis_column="wavenumber",
            response_column="absorbance",
            has_header=has_header,
        )


@pytest.mark.parametrize(
    "encoding",
    [
        1,
        True,
        ["utf-8"],
    ],
)
def test_read_spectrum_rejects_invalid_encoding_type(
    encoding,
):
    with pytest.raises(
        TypeError,
        match="encoding",
    ):
        read_spectrum(
            DATA_DIR / "example_spectrum.csv",
            axis_column="wavenumber",
            response_column="absorbance",
            encoding=encoding,
        )


def test_read_spectrum_rejects_unknown_encoding():
    with pytest.raises(
        ValueError,
        match="Unknown text encoding",
    ):
        read_spectrum(
            DATA_DIR / "example_spectrum.csv",
            axis_column="wavenumber",
            response_column="absorbance",
            encoding="not-a-real-encoding",
        )


@pytest.mark.parametrize(
    "comment_prefix",
    [
        1,
        True,
        ["#"],
    ],
)
def test_read_spectrum_rejects_invalid_comment_prefix_type(
    comment_prefix,
):
    with pytest.raises(
        TypeError,
        match="comment_prefix",
    ):
        read_spectrum(
            DATA_DIR / "example_spectrum.csv",
            axis_column="wavenumber",
            response_column="absorbance",
            comment_prefix=comment_prefix,
        )


@pytest.mark.parametrize(
    "comment_prefix",
    [
        "",
        "##",
        "\n",
        "\r",
        " ",
        "\t",
    ],
)
def test_read_spectrum_rejects_invalid_comment_prefix_value(
    comment_prefix,
):
    with pytest.raises(
        ValueError,
        match="comment_prefix",
    ):
        read_spectrum(
            DATA_DIR / "example_spectrum.csv",
            axis_column="wavenumber",
            response_column="absorbance",
            comment_prefix=comment_prefix,
        )


@pytest.mark.parametrize(
    "metadata",
    [
        [],
        [("note", "test")],
        "metadata",
        1,
    ],
)
def test_read_spectrum_rejects_nonmapping_metadata(
    metadata,
):
    with pytest.raises(
        TypeError,
        match="metadata",
    ):
        read_spectrum(
            DATA_DIR / "example_spectrum.csv",
            axis_column="wavenumber",
            response_column="absorbance",
            metadata=metadata,
        )


def test_read_spectrum_rejects_ambiguous_named_column(
    tmp_path,
):
    path = tmp_path / "duplicate_headers.csv"

    path.write_text(
        "signal,signal\n"
        "4000,0.10\n"
        "3000,0.20\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="ambiguous",
    ):
        read_spectrum(
            path,
            axis_column="signal",
            response_column=1,
        )


def test_read_spectrum_allows_position_for_duplicate_headers(
    tmp_path,
):
    path = tmp_path / "duplicate_headers.csv"

    path.write_text(
        "signal,signal\n"
        "4000,0.10\n"
        "3000,0.20\n",
        encoding="utf-8",
    )

    spectrum = read_spectrum(
        path,
        axis_column=0,
        response_column=1,
    )

    np.testing.assert_array_equal(
        spectrum.axis,
        [4000.0, 3000.0],
    )

    np.testing.assert_array_equal(
        spectrum.response,
        [0.10, 0.20],
    )