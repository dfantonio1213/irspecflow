# IRSpecFlow

Reproducible processing and analysis of infrared spectra.

IRSpecFlow is a Python package under development for importing, validating, preprocessing, aggregating, visualizing, and analyzing infrared spectra. It is intended for general infrared spectroscopy workflows, including quality control, construction of representative spectra, and preparation for multivariate analysis.

## Status

IRSpecFlow is in early development. The current implementation provides a `Spectrum` data model for one-dimensional infrared spectra and a generic importer for reading one local delimited-text spectrum file into a `Spectrum` object.

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

The `Spectrum` constructor checks only the requirements needed to store a spectrum consistently. It does not currently reject missing numerical values (`NaN`, meaning "not a number"), infinite values, duplicated axis values, non-monotonic ordering (an axis that does not consistently increase or decrease), empty or very short spectra, or unit declarations based on their scientific meaning.

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

Conditions such as `NaN`, positive or negative infinity, duplicate axis values, and non-monotonic ordering are preserved rather than rejected during import. A header-based table with the requested columns but no data rows is also accepted and produces an empty `Spectrum`. These structural conditions are left for later validation.

## Basic Usage

After cloning the repository and installing IRSpecFlow, the included synthetic CSV can be imported from the repository root with explicit column selection and scientific descriptors:

```python
from irspecflow.io import read_spectrum

spectrum = read_spectrum(
    "data/example_spectrum.csv",
    axis_column="wavenumber",
    response_column="absorbance",
    axis_unit="cm^-1",
    response_type="absorbance",
    spectrum_id="example-spectrum",
)

print(spectrum.axis)
print(spectrum.response)
```

The import record shows the source filename and parsing settings used to produce the `Spectrum`. The filename is recorded without storing its full filesystem path:

```python
print(spectrum.metadata["provenance"]["import"])
```

Caller-supplied metadata is not modified when the import record is added. Existing unrelated provenance information is preserved. If the metadata already contains an identical import record under `metadata["provenance"]["import"]`, it is accepted. If a different import record already exists there, the import fails rather than silently replacing it.

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

Import the function with `from irspecflow.io import read_spectrum`. `from irspecflow import read_spectrum` is not currently supported.

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

A runnable import example is available in `examples/basic_workflow.py`.

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

Future releases are planned to add:

- structural spectral validation;
- absorbance and transmittance conversion utilities;
- utilities for common spectral grids and interpolation;
- spectral visualization and quantitative quality control;
- baseline correction and Savitzky-Golay smoothing and derivatives;
- standard normal variate (SNV), multiplicative scatter correction (MSC), and comparison of preprocessing methods;
- hierarchical aggregation of spectra and representative-spectrum statistics;
- preparation of data for chemometric analysis with appropriate separation between model development and evaluation data.

## Repository Structure

```text
irspecflow/
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
│       └── validation.py
├── tests/
│   ├── test_import.py
│   └── test_spectrum.py
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