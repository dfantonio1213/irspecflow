"""Basic usage example for IRSpecFlow."""

from irspecflow import Spectrum


spectrum = Spectrum(
    axis=[
        4000.0,
        3000.0,
        2000.0,
        1000.0,
        400.0,
    ],
    response=[
        0.08,
        0.12,
        0.20,
        0.35,
        0.18,
    ],
    axis_unit="cm^-1",
    response_type="absorbance",
    spectrum_id="example-spectrum",
    metadata={
        "acquisition": {
            "resolution_cm1": 4,
        }
    },
)

print("Spectrum ID:", spectrum.spectrum_id)
print("Axis:", spectrum.axis)
print("Response:", spectrum.response)