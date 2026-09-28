# IRSpecFlow

Reproducible processing and analysis of infrared spectra.

IRSpecFlow is a Python package under development for importing, validating, preprocessing, aggregating, visualizing, and analyzing infrared spectra. The project is designed for general infrared spectroscopy workflows, including quality control, representative-spectrum construction, and preparation for multivariate analysis.

## Status

IRSpecFlow is in early development. The current release provides the package structure, dependency configuration, and test infrastructure needed for subsequent implementation.

Development toward version `1.0.0` will add:

- a consistent representation of spectral data and acquisition metadata;
- validation and quality-control tools for spectral datasets;
- parameterized preprocessing operations;
- support for technical replicates and hierarchical sample structures;
- aggregation and variability statistics for representative spectra;
- spectral visualization;
- preprocessing workflows for subsequent chemometric analysis.

Acquisition properties such as spectral range, numerical spacing, instrument resolution, scan count, replicate structure, and sample metadata will be represented explicitly in the data model or processing configuration. This design supports datasets collected under different experimental conditions.

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
│   └── test_import.py
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

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/Scripts/activate
```

Install IRSpecFlow and the development dependencies:

```bash
python -m pip install -e ".[dev]"
```

## Tests

Run the test suite with:

```bash
pytest
```

## Versioning

IRSpecFlow uses semantic versioning.

Releases before `1.0.0` represent active development of the initial API and may introduce incompatible interface changes. Version `1.0.0` will mark the first complete release of the initial package interface and documented processing workflow.