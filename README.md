# IRSpecFlow

Reproducible processing and analysis of infrared spectra.

IRSpecFlow is a Python package under development for importing, validating, preprocessing, aggregating, visualizing, and analyzing infrared spectra. It is intended for general infrared spectroscopy workflows, including quality control, construction of representative spectra, and preparation for multivariate analysis.

## Status

IRSpecFlow is in early development. The current implementation provides the `Spectrum` data model for one-dimensional infrared spectra and associated metadata.

The `Spectrum` interface supports:

- one-dimensional real-valued numeric arrays for the spectral axis and response;
- conversion of numerical input to independent NumPy `float64` arrays;
- read-only access to stored numerical data;
- optional spectral axis units and response types;
- optional spectrum identifiers;
- metadata for user-defined sampling hierarchies;
- acquisition, sample, instrument, provenance, and additional metadata;
- package-level import with `from irspecflow import Spectrum`.

The `Spectrum` constructor enforces only the invariants required to store a spectrum consistently. The planned validation layer will handle structural checks for missing or non-finite values, duplicated axis values, axis monotonicity, recognized units, and plausible spectrum length.

Planned features include:

- generic text and CSV import with canonicalization support;
- structural validation and quality-control tools;
- absorbance and transmittance conversion utilities;
- utilities for common spectral grids and interpolation;
- spectral visualization;
- baseline correction and Savitzky-Golay smoothing and derivatives;
- SNV, MSC, and comparison of preprocessing methods;
- hierarchical aggregation of spectra and representative-spectrum statistics;
- leakage-safe preparation for chemometric analysis.

## Basic Usage

```python
from irspecflow import Spectrum

spectrum = Spectrum(
    axis=[4000.0, 3000.0, 2000.0, 1000.0, 400.0],
    response=[0.08, 0.12, 0.20, 0.35, 0.18],
    axis_unit="cm^-1",
    response_type="absorbance",
    spectrum_id="example-spectrum",
    metadata={
        "acquisition": {
            "resolution_cm1": 4,
        }
    },
)

print(spectrum.axis)
print(spectrum.response)
```

The supplied axis ordering is preserved. Unit conversion, axis canonicalization, interpolation, and structural validation are not yet implemented. These operations are planned for later releases.

A runnable version of this example is available in `examples/basic_workflow.py`.

## Repository Structure

```text
irspecflow/
├── data/
│   └── README.md
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

During pre-`1.0` development, each minor release corresponds to a planned set of features. Patch releases are reserved for fixes within a minor release. The public API may change before `1.0.0` as new functionality is implemented and tested.

Version `1.0.0` is planned as the first stable, tested, and documented public API.