"""Basic usage example for IRSpecFlow."""

from pathlib import Path

from irspecflow.io import read_spectrum
from irspecflow.validation import validate_spectrum


data_file = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "example_spectrum.csv"
)

spectrum = read_spectrum(
    data_file,
    axis_column="wavenumber",
    response_column="absorbance",
    axis_unit="cm^-1",
    response_type="absorbance",
    spectrum_id="example-spectrum",
    metadata={
        "acquisition": {
            "resolution_cm1": 4,
        }
    },
)

findings = validate_spectrum(spectrum)

print("Spectrum ID:", spectrum.spectrum_id)
print("Axis:", spectrum.axis)
print("Response:", spectrum.response)
print(
    "Source file:",
    spectrum.metadata["provenance"]["import"]["source_filename"],
)
print("Validation findings:", findings)