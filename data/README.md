# Data

This directory contains example datasets and test fixtures used by IRSpecFlow documentation, examples, and automated tests.

The persistent synthetic files currently included are:

- `example_spectrum.csv`: a small comma-delimited spectrum table with a header row and an unused text column;
- `example_spectrum_headerless.tsv`: a small headerless tab-delimited spectrum table.

These files contain only synthetic example data. They are not experimental measurements.

Most deliberately invalid or unusual inputs used for testing are created temporarily by the automated tests instead of being stored in this directory.

Source spectral files should be retained unchanged when they are used as inputs to reproducible workflows. Derived datasets should be stored separately or generated reproducibly from the corresponding source data and processing configuration.

Datasets containing sensitive information or files unsuitable for normal Git storage should be managed using an appropriate external storage system.