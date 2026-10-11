"""Visualization and quality-control usage example for IRSpecFlow."""

from irspecflow import Spectrum
from irspecflow.plotting import plot_spectra, plot_spectrum
from irspecflow.qc import response_range


spectrum_a = Spectrum(
    axis=[4000.0, 3200.0, 2400.0, 1600.0, 800.0, 400.0],
    response=[0.125, 0.1875, 0.25, 0.3125, 0.375, 0.5],
    axis_unit="cm^-1",
    response_type="absorbance",
    spectrum_id="example-a",
)

spectrum_b = Spectrum(
    axis=[4000.0, 3000.0, 2000.0, 1000.0, 400.0],
    response=[0.15625, 0.21875, 0.28125, 0.34375, 0.40625],
    axis_unit="cm^-1",
    response_type="absorbance",
    spectrum_id="example-b",
)

print(
    "Response range (example-a):",
    response_range(spectrum_a),
)
print(
    "Response range (example-b):",
    response_range(spectrum_b),
)

single_axes = plot_spectrum(spectrum_a)
single_axes.set_title("Single spectrum")

overlay_axes = plot_spectra([spectrum_a, spectrum_b])
overlay_axes.set_title("Spectrum overlay")
overlay_axes.legend()