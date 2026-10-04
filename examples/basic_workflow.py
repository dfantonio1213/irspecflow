"""Basic usage example for IRSpecFlow."""

from pathlib import Path

from irspecflow import Spectrum
from irspecflow.io import read_spectrum
from irspecflow.transformations import (
    interpolate_spectrum,
    transmittance_to_absorbance,
)
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

interpolated = interpolate_spectrum(
    spectrum,
    [4000.0, 2500.0, 1000.0, 400.0],
)

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

print("Spectrum ID:", spectrum.spectrum_id)
print("Axis:", spectrum.axis)
print("Response:", spectrum.response)
print(
    "Source file:",
    spectrum.metadata["provenance"]["import"]["source_filename"],
)
print("Validation findings:", findings)

print("Interpolated axis:", interpolated.axis)
print("Interpolated response:", interpolated.response)
print(
    "Interpolation provenance:",
    interpolated.metadata["provenance"]["transformations"],
)

print("Converted absorbance:", converted.response)
print(
    "Conversion provenance:",
    converted.metadata["provenance"]["transformations"],
)