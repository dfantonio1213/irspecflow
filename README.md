# IRSpecFlow

Reproducible processing and analysis of infrared spectra.

IRSpecFlow is a Python package under development for importing, validating, preprocessing, aggregating, visualizing, and analyzing infrared spectra. It is intended for general infrared spectroscopy workflows, including quality control, construction of representative spectra, and preparation for multivariate analysis.

## Status

IRSpecFlow is in early development. The current implementation provides a `Spectrum` data model for one-dimensional infrared spectra, a generic importer for reading one local delimited-text spectrum file into a `Spectrum`, structural validation for defined conditions observable from one stored spectrum, and explicit single-spectrum transformations for percent-transmittance-to-absorbance conversion and linear interpolation to a caller-supplied target axis.

### Spectrum model

In practical terms, a `Spectrum` stores a spectral axis, such as wavenumber, together with the corresponding measured response values, such as absorbance or transmittance.

The `Spectrum` interface supports:

- one-dimensional real-valued numeric arrays for the spectral axis and response;
- conversion of numerical input to independent NumPy `float64` (64-bit floating-point) arrays;
- read-only access to stored numerical data;
- optional spectral axis units, response types, and response units;
- optional spectrum identifiers;
- user-defined sampling hierarchies for organizing related spectra;
- acquisition, sample, instrument, provenance, and other metadata;
- package-level import with `from irspecflow import Spectrum`.

The `Spectrum` constructor checks only the requirements needed to store a spectrum consistently. It permits structural conditions such as missing numerical values (`NaN`, meaning "not a number"), infinite values, duplicated axis values, non-monotonic ordering (an axis that does not consistently increase or decrease), empty spectra, and unrecognized scientific declarations when they are otherwise representable. These conditions can be inspected separately with `irspecflow.validation.validate_spectrum()` rather than being rejected or corrected during storage.

### Delimited-text import

A delimited-text file is a table in which columns are separated by a consistent character, such as a comma, tab, semicolon, or pipe.

`irspecflow.io.read_spectrum()` reads one local delimited-text file and returns one `Spectrum`.

The importer supports:

- a local file path supplied as a string, such as `"data/spectrum.csv"`, or as a path-like object such as `pathlib.Path`;
- explicit axis and response selection by exact, case-sensitive column name or by column position, where `0` is the first column, `1` is the second, and so on;
- text or other non-numerical values in unselected columns, provided the file can still be parsed as a valid delimited table; only the selected axis and response columns must be numerically representable;
- positional column selection for files without a header row;
- files whose first parsed table row contains column names, as well as files with no header row;
- a comma delimiter by default or another explicit single-character delimiter;
- UTF-8 by default or another explicitly selected text encoding;
- an optional single non-whitespace `comment_prefix` character for leading comment or metadata lines;
- explicitly supplied axis and response units, response type, spectrum identifier, sampling hierarchy, and other metadata;
- preservation of parsed row order and conversion of selected numerical values to the `Spectrum` `float64` representation without scientific transformation;
- an import record describing the importer, source filename, selected columns, and parsing settings. This record is stored in `metadata["provenance"]["import"]`.

If a requested column name appears more than once in the header, select that column by position instead.

Some instrument exports begin with descriptive lines before the numerical table. The optional `comment_prefix` setting can be used to skip these leading lines when their first non-whitespace character is a consistent marker such as `#` or `;`. If `comment_prefix` is not supplied, no prefix-based comment or metadata lines are skipped automatically. Leading whitespace before the character is allowed, and blank lines within this introductory block are also skipped. For example, `comment_prefix="#"` can be used for exports that begin with `#` metadata lines, and `comment_prefix=";"` can be used for analogous semicolon-prefixed metadata. Lines beginning with the same character after the table has started are treated as table content rather than comments. The skipped lines are ignored; their contents are not automatically added to the `Spectrum` metadata.

An empty field in the selected axis or response column is retained as `NaN`. Text such as `N/A`, `NULL`, or `missing` is not automatically treated as missing numerical data and will cause the import to fail if it appears in either selected column.

The importer does not automatically detect delimiters, whether a header is present, text encodings, units, whether the response represents absorbance or transmittance, or other scientific meaning from filenames, column names, or numerical values. It assumes the first parsed table row is a header by default; use `has_header=False` for a headerless file.

Import is separate from structural validation and spectral processing. `read_spectrum()` preserves parsed row order and does not intentionally alter the selected numerical values beyond conversion to the `Spectrum` `float64` representation. It does not sort or reverse the spectral axis, remove duplicate values, reject non-monotonic spectra, convert between units or response quantities, interpolate, smooth, correct baselines, or calculate quality-control metrics. It also does not scan directories, import batches of files, or provide dedicated readers for proprietary or vendor-specific instrument formats.

Conditions such as `NaN`, positive or negative infinity, duplicate axis values, and non-monotonic ordering are preserved rather than rejected during import. A header-based table with the requested columns but no data rows is also accepted and produces an empty `Spectrum`. These conditions remain in the imported `Spectrum` so structural validation can report them without modifying the imported data.

### Structural validation

`irspecflow.validation.validate_spectrum()` inspects one `Spectrum` and reports defined structural conditions without modifying the stored spectrum. It returns a tuple of `ValidationFinding` objects in a consistent, deterministic order. Each finding contains a stable machine-readable `code` and a human-readable `message`.

Structural validation checks for:

- empty spectra;
- `NaN`, positive infinity, and negative infinity in the spectral axis and response as separate conditions;
- exactly duplicated finite axis values;
- genuine reversals in the direction of a fully finite spectral axis;
- unrecognized axis-unit, response-type, and response-unit declarations;
- the inconsistent combination of an absorbance response declared with the `%` response unit.

Strictly increasing and strictly decreasing finite axes are both accepted. Repeated axis values are reported separately from direction reversals. Duplicate detection uses exact stored `float64` values; near-equal values are not treated as duplicates. If the axis contains a non-finite value, the validator reports the relevant non-finite condition but does not make a monotonicity judgment.

An empty spectrum produces an `empty_spectrum` finding. A one-point spectrum can have no findings when its axis and response values are finite and its supplied declarations are recognized or omitted and mutually consistent where applicable. Two distinct finite axis values are accepted whether they increase or decrease. The validator does not impose a universal minimum number of spectral points beyond detecting an empty spectrum.

Scientific declarations remain optional. When supplied, recognition is exact and case-sensitive: `cm^-1` is the recognized axis unit, `absorbance` and `transmittance` are the recognized response types, and `%` is the recognized supplied response unit. The combination `response_type="transmittance"` with `response_unit="%"` is accepted, while `response_type="absorbance"` with `response_unit="%"` produces an inconsistency finding. Alternate spellings, capitalization, Unicode notation, aliases, and surrounding whitespace are not normalized automatically.

A spectrum with no detected structural findings returns an empty tuple, `()`. This means only that none of the structural conditions defined by this validator were detected. It does not establish measurement quality or determine whether the spectrum is scientifically suitable for a particular analysis.

Structural conditions in a legitimate `Spectrum` are returned as findings. Passing an object that is not a `Spectrum` is instead an API usage error and raises `TypeError`.

Validation is inspection-only. It does not sort or reverse the axis, remove duplicate or non-finite values, convert axis or response units, convert between absorbance and transmittance, interpolate, repair spectra, perform preprocessing, or calculate measurement-quality or replicate-level quality-control metrics.

### Spectral transformations

`irspecflow.transformations` provides two explicit transformations for one `Spectrum`:

- percent-transmittance-to-absorbance conversion with `transmittance_to_absorbance()`;
- linear interpolation to an explicit caller-supplied target axis with `interpolate_spectrum()`.

Both operations return new ordinary `Spectrum` objects and leave the supplied source spectrum unchanged. `Spectrum` subclasses are accepted as inputs, but the returned object uses the ordinary `Spectrum` representation.

#### Percent transmittance to absorbance

`transmittance_to_absorbance()` converts percent transmittance to absorbance according to:

```text
A = -log10(T / 100)
```

where `T` is the percent-transmittance value.

The input spectrum must:

- contain at least one spectral point;
- declare `response_type="transmittance"` exactly;
- declare `response_unit="%"` exactly;
- contain only finite response values;
- contain response values strictly greater than zero.

Missing, alternate, or incompatible response declarations are not inferred or normalized.

Values above `100 %T` remain mathematically valid inputs. They may produce negative absorbance values, but IRSpecFlow does not treat that condition alone as a transformation failure. Scientific plausibility and measurement-quality judgments remain separate quality-control concerns.

Structural conditions limited to the spectral axis do not prevent response conversion because the calculation operates only on response values. The spectral axis is preserved exactly as stored.

The returned spectrum has:

- the converted absorbance response;
- `response_type="absorbance"`;
- `response_unit=None`;
- the original axis values and order;
- the original axis unit;
- the original spectrum identifier;
- preserved hierarchy and unrelated metadata;
- preserved import provenance and earlier transformation provenance.

#### Linear interpolation

`interpolate_spectrum()` linearly interpolates one spectrum onto an explicit `target_axis`.

The source spectrum must contain:

- at least two spectral points;
- finite axis values;
- finite response values;
- no duplicate finite axis values;
- an axis that is consistently increasing or consistently decreasing.

Scientific declaration findings do not prevent interpolation when the stored numerical values otherwise satisfy these requirements. The operation does not infer or convert scientific units.

The target axis must:

- be one-dimensional;
- contain real numeric values;
- contain at least one value;
- contain only finite values;
- contain no exact duplicate values after numerical preparation.

A target containing two or more points must be strictly increasing or strictly decreasing. A one-point target is valid.

Every target value must lie within the closed numerical range of the source axis. Exact source endpoints are accepted. Values outside the source range are rejected rather than extrapolated or clipped.

Increasing and decreasing source axes are both supported. Increasing and decreasing target axes are also supported, and the returned `Spectrum` preserves the caller's requested target order.

`target_axis` is assumed to represent the same physical coordinate system and unit as the source axis. Interpolation does not perform axis-unit conversion.

The operation does not extrapolate, clip values to the source range, sort the caller's target axis, remove duplicates, or otherwise repair the source or target data.

Interpolating onto a target axis numerically identical to the source axis still returns a distinct new `Spectrum` and records the interpolation.

#### Transformation provenance

Each successful transformation appends one record to:

```python
metadata["provenance"]["transformations"]
```

The value is an ordered list, so transformation order is retained across chained operations.

Percent-transmittance-to-absorbance conversion records:

```python
{
    "operation": (
        "irspecflow.transformations."
        "transmittance_to_absorbance"
    )
}
```

Linear interpolation records:

```python
{
    "operation": (
        "irspecflow.transformations."
        "interpolate_spectrum"
    ),
    "method": "linear",
}
```

Existing import provenance, unrelated provenance information, unrelated metadata, and previous transformation entries are preserved. Transformation metadata handling does not modify the source spectrum's metadata in place.

The full target axis and numerical response arrays are not duplicated in provenance because the returned `Spectrum` already contains the transformed numerical data.

Import the transformation functions from their module:

```python
from irspecflow.transformations import (
    interpolate_spectrum,
    transmittance_to_absorbance,
)
```

The transformation functions are not re-exported from the package root.

## Basic Usage

After cloning the repository and installing IRSpecFlow, the included synthetic absorbance CSV can be imported from the repository root with explicit column selection and scientific descriptors:

```python
from irspecflow import Spectrum
from irspecflow.io import read_spectrum
from irspecflow.transformations import (
    interpolate_spectrum,
    transmittance_to_absorbance,
)
from irspecflow.validation import validate_spectrum

spectrum = read_spectrum(
    "data/example_spectrum.csv",
    axis_column="wavenumber",
    response_column="absorbance",
    axis_unit="cm^-1",
    response_type="absorbance",
    spectrum_id="example-spectrum",
)

findings = validate_spectrum(spectrum)

print(spectrum.axis)
print(spectrum.response)
print(findings)
```

For this included example, `findings` is an empty tuple because none of the structural conditions checked by the validator are present.

The imported absorbance spectrum can then be interpolated explicitly to an in-range target axis:

```python
interpolated = interpolate_spectrum(
    spectrum,
    [4000.0, 2500.0, 1000.0, 400.0],
)

print(interpolated.axis)
print(interpolated.response)
print(
    interpolated.metadata["provenance"]["transformations"]
)
```

The interpolation returns a new `Spectrum`; the imported source remains unchanged. Its import provenance is preserved in the transformed result.

The import record for the original spectrum shows the source filename and parsing settings used to construct it. The filename is recorded without storing its full filesystem path:

```python
print(spectrum.metadata["provenance"]["import"])
```

Percent-transmittance-to-absorbance conversion can be demonstrated with a directly constructed spectrum:

```python
transmittance_spectrum = Spectrum(
    axis=[4000.0, 2500.0, 1000.0],
    response=[100.0, 50.0, 10.0],
    axis_unit="cm^-1",
    response_type="transmittance",
    response_unit="%",
    spectrum_id="example-transmittance",
)

converted = transmittance_to_absorbance(
    transmittance_spectrum
)

print(converted.response)
print(converted.response_type)
print(converted.response_unit)
print(
    converted.metadata["provenance"]["transformations"]
)
```

The original percent-transmittance spectrum remains unchanged.

Caller-supplied metadata is not modified when import or transformation provenance is added. Existing unrelated provenance information is preserved.

For a file that begins with comment or metadata lines, supply the relevant prefix explicitly:

```python
spectrum = read_spectrum(
    "instrument_export.txt",
    axis_column="wavenumber",
    response_column="absorbance",
    comment_prefix="#",
)
```

For a file without a header row, select columns by position and set `has_header=False`. For example, the included tab-delimited file can be read with:

```python
spectrum = read_spectrum(
    "data/example_spectrum_headerless.tsv",
    axis_column=0,
    response_column=1,
    delimiter="\t",
    has_header=False,
)
```

Import `read_spectrum()` from `irspecflow.io` and the transformation functions from `irspecflow.transformations`. These functions are not currently re-exported from the package root.

Direct construction of a `Spectrum` also remains supported:

```python
from irspecflow import Spectrum

spectrum = Spectrum(
    axis=[4000.0, 3000.0, 2000.0],
    response=[0.08, 0.12, 0.20],
    axis_unit="cm^-1",
    response_type="absorbance",
)
```

A runnable import, validation, interpolation, and response-conversion example is available in `examples/basic_workflow.py`.

## Current Import Scope

The current importer reads one local delimited-text file and produces one `Spectrum`.

The filename extension does not determine whether a file can be imported. Files such as `.csv`, `.txt`, or `.dat` can be read when they contain a supported delimited-text table and the correct import settings are supplied.

The current import interface does not provide:

- files accessed directly from web URLs or remote storage;
- already-open file objects or in-memory buffers;
- compressed files or archives;
- automatic detection of delimiters, headers, or text encoding;
- advanced table-reading options beyond the documented settings above;
- multi-row headers or selecting an arbitrary row as the header; after any requested leading comment or metadata lines are skipped, the first table row is treated as the header unless `has_header=False`;
- multi-character or pattern-based delimiters;
- importing an entire directory or batch of files at once;
- extracting multiple spectra from one table;
- dedicated readers for proprietary or vendor-specific instrument formats.

## Planned Features

Future development may add:

- additional response-conversion directions and spectral-axis unit support;
- collection-level common-grid construction and batch or dataset transformation;
- spectral visualization and quantitative quality control;
- baseline correction and Savitzky-Golay smoothing and derivatives;
- standard normal variate (SNV), multiplicative scatter correction (MSC), and comparison of preprocessing methods;
- hierarchical aggregation of spectra and representative-spectrum statistics;
- preparation of data for chemometric analysis with appropriate separation between model development and evaluation data.

## Repository Structure

```text
irspecflow/
├── .github/
│   └── workflows/
│       └── tests.yml
├── data/
│   ├── README.md
│   ├── example_spectrum.csv
│   └── example_spectrum_headerless.tsv
├── examples/
│   └── basic_workflow.py
├── src/
│   └── irspecflow/
│       ├── __init__.py
│       ├── aggregation.py
│       ├── io.py
│       ├── plotting.py
│       ├── preprocessing.py
│       ├── qc.py
│       ├── spectrum.py
│       ├── transformations.py
│       └── validation.py
├── tests/
│   ├── test_import.py
│   ├── test_spectrum.py
│   ├── test_transformations.py
│   └── test_validation.py
├── .gitignore
├── README.md
└── pyproject.toml
```

## Development Installation

Clone the repository:

```bash
git clone https://github.com/dfantonio1213/irspecflow.git
cd irspecflow
```

IRSpecFlow requires Python 3.11 or later. Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

On Git Bash:

```bash
source .venv/Scripts/activate
```

Install IRSpecFlow and the development dependencies:

```bash
python -m pip install -e ".[dev]"
```

## Tests

Run the test suite with:

```bash
python -m pytest
```

## Versioning

IRSpecFlow uses semantic versioning.

Before version `1.0.0`, minor releases may add or revise public functionality, while patch releases are used for fixes. The public API may therefore change during the pre-`1.0` development period.

Version `1.0.0` is planned as the first stable, tested, and documented public API.