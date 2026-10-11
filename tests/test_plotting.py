from copy import deepcopy
from inspect import Parameter, signature

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pytest

import irspecflow
import irspecflow.plotting
from irspecflow import Spectrum
from irspecflow.plotting import plot_spectra, plot_spectrum


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _spectrum(
    axis=(4000.0, 3000.0, 2000.0),
    response=(0.10, 0.20, 0.30),
    **kwargs,
):
    return Spectrum(
        axis,
        response,
        **kwargs,
    )


def test_plotting_public_interface_and_signatures():
    assert irspecflow.plotting.plot_spectrum is plot_spectrum
    assert irspecflow.plotting.plot_spectra is plot_spectra
    assert irspecflow.plotting.__all__ == [
        "plot_spectrum",
        "plot_spectra",
    ]
    assert not hasattr(irspecflow, "plot_spectrum")
    assert not hasattr(irspecflow, "plot_spectra")

    single_parameters = signature(plot_spectrum).parameters
    overlay_parameters = signature(plot_spectra).parameters

    assert list(single_parameters) == [
        "spectrum",
        "ax",
        "label",
        "xlabel",
        "ylabel",
        "axis_descending",
    ]
    assert list(overlay_parameters) == [
        "spectra",
        "ax",
        "labels",
        "xlabel",
        "ylabel",
        "axis_descending",
    ]
    assert (
        single_parameters["spectrum"].kind
        is Parameter.POSITIONAL_OR_KEYWORD
    )
    assert (
        overlay_parameters["spectra"].kind
        is Parameter.POSITIONAL_OR_KEYWORD
    )

    for name in [
        "ax",
        "label",
        "xlabel",
        "ylabel",
        "axis_descending",
    ]:
        assert single_parameters[name].kind is Parameter.KEYWORD_ONLY

    for name in [
        "ax",
        "labels",
        "xlabel",
        "ylabel",
        "axis_descending",
    ]:
        assert overlay_parameters[name].kind is Parameter.KEYWORD_ONLY

    assert single_parameters["spectrum"].default is Parameter.empty
    assert overlay_parameters["spectra"].default is Parameter.empty

    for name in [
        "ax",
        "label",
        "xlabel",
        "ylabel",
        "axis_descending",
    ]:
        assert single_parameters[name].default is None

    for name in [
        "ax",
        "labels",
        "xlabel",
        "ylabel",
        "axis_descending",
    ]:
        assert overlay_parameters[name].default is None


def test_plotting_accepts_subclasses_and_returns_axes():
    class DerivedSpectrum(Spectrum):
        pass

    first = DerivedSpectrum(
        [4000.0, 3000.0],
        [0.10, 0.20],
    )
    second = DerivedSpectrum(
        [3900.0, 2500.0, 1000.0],
        [0.15, 0.25, 0.35],
    )

    single_ax = plot_spectrum(first)
    overlay_ax = plot_spectra([first, second])

    assert isinstance(single_ax, Axes)
    assert isinstance(overlay_ax, Axes)
    assert len(single_ax.lines) == 1
    assert len(overlay_ax.lines) == 2


def test_plotting_reuses_exact_supplied_axes():
    _, single_ax = plt.subplots()
    _, overlay_ax = plt.subplots()

    figures_before = set(
        plt.get_fignums()
    )

    single_result = plot_spectrum(
        _spectrum(),
        ax=single_ax,
    )
    overlay_result = plot_spectra(
        [_spectrum(), _spectrum()],
        ax=overlay_ax,
    )

    assert single_result is single_ax
    assert overlay_result is overlay_ax
    assert set(plt.get_fignums()) == figures_before


def test_plot_spectrum_uses_exact_stored_data_without_reordering():
    spectrum = _spectrum(
        axis=(3000.0, 1000.0, 2000.0),
        response=(0.30, 0.10, 0.20),
        axis_unit="cm^-1",
    )
    axis_before = spectrum.axis.copy()
    response_before = spectrum.response.copy()

    ax = plot_spectrum(spectrum)

    np.testing.assert_array_equal(
        ax.lines[0].get_xdata(),
        axis_before,
    )
    np.testing.assert_array_equal(
        ax.lines[0].get_ydata(),
        response_before,
    )
    np.testing.assert_array_equal(
        spectrum.axis,
        axis_before,
    )
    np.testing.assert_array_equal(
        spectrum.response,
        response_before,
    )


@pytest.mark.parametrize(
    ("axis_unit", "expected_descending"),
    [
        ("cm^-1", True),
        (None, False),
        ("cm-1", False),
    ],
)
def test_plot_spectrum_automatic_direction(
    axis_unit,
    expected_descending,
):
    ax = plot_spectrum(
        _spectrum(
            axis=(1000.0, 2000.0, 3000.0),
            axis_unit=axis_unit,
        )
    )

    assert bool(ax.xaxis_inverted()) == expected_descending


@pytest.mark.parametrize(
    ("axis_unit", "axis_descending", "expected_descending"),
    [
        ("cm^-1", False, False),
        (None, True, True),
        ("cm-1", True, True),
        ("cm^-1", True, True),
    ],
)
def test_plot_spectrum_explicit_direction_override(
    axis_unit,
    axis_descending,
    expected_descending,
):
    ax = plot_spectrum(
        _spectrum(axis_unit=axis_unit),
        axis_descending=axis_descending,
    )

    assert bool(ax.xaxis_inverted()) == expected_descending


def test_direction_is_idempotent_on_reused_axes():
    _, ax = plt.subplots()
    spectrum = _spectrum()

    for _ in range(2):
        plot_spectrum(
            spectrum,
            ax=ax,
            axis_descending=True,
        )

    assert ax.xaxis_inverted()

    for _ in range(2):
        plot_spectrum(
            spectrum,
            ax=ax,
            axis_descending=False,
        )

    assert not ax.xaxis_inverted()


def test_plot_spectra_direction_is_idempotent_on_reused_axes():
    _, ax = plt.subplots()
    spectra = [
        _spectrum(),
        _spectrum(),
    ]

    for _ in range(2):
        plot_spectra(
            spectra,
            ax=ax,
            axis_descending=True,
        )

    assert ax.xaxis_inverted()

    for _ in range(2):
        plot_spectra(
            spectra,
            ax=ax,
            axis_descending=False,
        )

    assert not ax.xaxis_inverted()


@pytest.mark.parametrize(
    ("axis_unit", "expected"),
    [
        ("cm^-1", "Wavenumber (cm^-1)"),
        (None, "Axis"),
        ("cm-1", "Axis"),
    ],
)
def test_plot_spectrum_default_x_label(
    axis_unit,
    expected,
):
    ax = plot_spectrum(
        _spectrum(axis_unit=axis_unit)
    )

    assert ax.get_xlabel() == expected


@pytest.mark.parametrize(
    ("response_type", "response_unit", "expected"),
    [
        ("absorbance", None, "Absorbance"),
        ("transmittance", "%", "Transmittance (%)"),
        ("transmittance", None, "Transmittance"),
        (None, None, "Response"),
        ("reflectance", None, "Response"),
        ("absorbance", "%", "Response"),
        ("transmittance", "percent", "Response"),
    ],
)
def test_plot_spectrum_default_y_label(
    response_type,
    response_unit,
    expected,
):
    ax = plot_spectrum(
        _spectrum(
            response_type=response_type,
            response_unit=response_unit,
        )
    )

    assert ax.get_ylabel() == expected


def test_plot_spectrum_explicit_labels_and_spectrum_id_behavior():
    explicit = plot_spectrum(
        _spectrum(
            axis_unit="cm^-1",
            response_type="absorbance",
            spectrum_id="stored-id",
        ),
        label="explicit-label",
        xlabel="Custom x",
        ylabel="Custom y",
    )
    explicit_empty = plot_spectrum(
        _spectrum(spectrum_id="stored-id"),
        label="",
    )
    fallback = plot_spectrum(
        _spectrum(spectrum_id="stored-id")
    )
    unlabeled = plot_spectrum(
        _spectrum()
    )

    assert explicit.lines[0].get_label() == "explicit-label"
    assert explicit_empty.lines[0].get_label() == ""
    assert explicit.get_xlabel() == "Custom x"
    assert explicit.get_ylabel() == "Custom y"
    assert fallback.lines[0].get_label() == "stored-id"

    unlabeled_label = unlabeled.lines[0].get_label()

    assert (
        not unlabeled_label
        or unlabeled_label.startswith("_")
    )


def test_plot_spectra_allows_different_lengths_and_grids():
    first = _spectrum(
        axis=(4000.0, 3000.0, 2000.0),
        response=(0.10, 0.20, 0.30),
    )
    second = _spectrum(
        axis=(3900.0, 2500.0),
        response=(0.15, 0.25),
    )

    ax = plot_spectra([first, second])

    np.testing.assert_array_equal(
        ax.lines[0].get_xdata(),
        first.axis,
    )
    np.testing.assert_array_equal(
        ax.lines[0].get_ydata(),
        first.response,
    )
    np.testing.assert_array_equal(
        ax.lines[1].get_xdata(),
        second.axis,
    )
    np.testing.assert_array_equal(
        ax.lines[1].get_ydata(),
        second.response,
    )


@pytest.mark.parametrize(
    (
        "spectra",
        "expected_xlabel",
        "expected_ylabel",
        "expected_descending",
    ),
    [
        (
            [
                _spectrum(
                    axis_unit="cm^-1",
                    response_type="absorbance",
                ),
                _spectrum(
                    axis_unit="cm^-1",
                    response_type="absorbance",
                ),
            ],
            "Wavenumber (cm^-1)",
            "Absorbance",
            True,
        ),
        (
            [
                _spectrum(
                    axis_unit="cm^-1",
                    response_type="absorbance",
                ),
                _spectrum(
                    response_type="absorbance",
                ),
            ],
            "Axis",
            "Absorbance",
            False,
        ),
        (
            [
                _spectrum(
                    axis_unit="cm^-1",
                    response_type="absorbance",
                ),
                _spectrum(
                    axis_unit="cm^-1",
                    response_type="transmittance",
                    response_unit="%",
                ),
            ],
            "Wavenumber (cm^-1)",
            "Response",
            True,
        ),
        (
            [
                _spectrum(
                    response_type="transmittance",
                    response_unit="%",
                ),
                _spectrum(
                    response_type="transmittance",
                    response_unit="%",
                ),
            ],
            "Axis",
            "Transmittance (%)",
            False,
        ),
        (
            [
                _spectrum(
                    response_type="transmittance",
                ),
                _spectrum(
                    response_type="transmittance",
                ),
            ],
            "Axis",
            "Transmittance",
            False,
        ),
        (
            [
                _spectrum(
                    response_type="absorbance",
                ),
                _spectrum(
                    response_type="absorbance",
                    response_unit="%",
                ),
            ],
            "Axis",
            "Response",
            False,
        ),
    ],
)
def test_plot_spectra_conservative_labels_and_direction(
    spectra,
    expected_xlabel,
    expected_ylabel,
    expected_descending,
):
    ax = plot_spectra(spectra)

    assert ax.get_xlabel() == expected_xlabel
    assert ax.get_ylabel() == expected_ylabel
    assert bool(ax.xaxis_inverted()) == expected_descending


@pytest.mark.parametrize(
    ("axis_descending", "expected_descending"),
    [
        (True, True),
        (False, False),
    ],
)
def test_plot_spectra_explicit_direction_override(
    axis_descending,
    expected_descending,
):
    ax = plot_spectra(
        [
            _spectrum(axis_unit="cm^-1"),
            _spectrum(axis_unit=None),
        ],
        axis_descending=axis_descending,
    )

    assert bool(ax.xaxis_inverted()) == expected_descending


def test_plot_spectra_explicit_labels_and_overrides():
    first = _spectrum(
        axis_unit="cm^-1",
        response_type="absorbance",
        spectrum_id="stored-first",
    )
    second = _spectrum(
        axis_unit="cm^-1",
        response_type="absorbance",
        spectrum_id="stored-second",
    )

    ax = plot_spectra(
        [first, second],
        labels=["", "second"],
        xlabel="Custom x",
        ylabel="Custom y",
        axis_descending=False,
    )

    assert [
        line.get_label()
        for line in ax.lines
    ] == [
        "",
        "second",
    ]
    assert ax.get_xlabel() == "Custom x"
    assert ax.get_ylabel() == "Custom y"
    assert not ax.xaxis_inverted()


def test_plot_spectra_uses_spectrum_ids_when_labels_are_omitted():
    ax = plot_spectra(
        [
            _spectrum(
                spectrum_id="first",
            ),
            _spectrum(
                spectrum_id="second",
            ),
        ]
    )

    assert [
        line.get_label()
        for line in ax.lines
    ] == [
        "first",
        "second",
    ]


def test_plotting_does_not_create_title_or_legend_automatically():
    single_ax = plot_spectrum(
        _spectrum(
            spectrum_id="single",
        )
    )
    overlay_ax = plot_spectra(
        [
            _spectrum(
                spectrum_id="first",
            ),
            _spectrum(
                spectrum_id="second",
            ),
        ]
    )

    assert single_ax.get_title() == ""
    assert overlay_ax.get_title() == ""
    assert single_ax.get_legend() is None
    assert overlay_ax.get_legend() is None


@pytest.mark.parametrize(
    "spectrum",
    [
        None,
        [1.0, 2.0],
        {
            "axis": [
                1.0,
                2.0,
            ]
        },
    ],
)
def test_plot_spectrum_rejects_unsupported_input(
    spectrum,
):
    with pytest.raises(
        TypeError,
        match="spectrum must be a Spectrum",
    ):
        plot_spectrum(spectrum)


@pytest.mark.parametrize(
    "operation",
    [
        lambda value: plot_spectrum(
            _spectrum(),
            ax=value,
        ),
        lambda value: plot_spectra(
            [_spectrum()],
            ax=value,
        ),
    ],
)
def test_plotting_rejects_invalid_axes(
    operation,
):
    with pytest.raises(
        TypeError,
        match="ax must be a matplotlib Axes",
    ):
        operation(object())


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (
            lambda: plot_spectrum(
                _spectrum(),
                label=1,
            ),
            "label must be a string or None",
        ),
        (
            lambda: plot_spectrum(
                _spectrum(),
                xlabel=1,
            ),
            "xlabel must be a string or None",
        ),
        (
            lambda: plot_spectrum(
                _spectrum(),
                ylabel=[],
            ),
            "ylabel must be a string or None",
        ),
        (
            lambda: plot_spectra(
                [_spectrum()],
                xlabel=1,
            ),
            "xlabel must be a string or None",
        ),
        (
            lambda: plot_spectra(
                [_spectrum()],
                ylabel=[],
            ),
            "ylabel must be a string or None",
        ),
    ],
)
def test_plotting_rejects_nonstring_explicit_labels(
    operation,
    message,
):
    with pytest.raises(
        TypeError,
        match=message,
    ):
        operation()


@pytest.mark.parametrize(
    "operation",
    [
        lambda value: plot_spectrum(
            _spectrum(),
            axis_descending=value,
        ),
        lambda value: plot_spectra(
            [_spectrum()],
            axis_descending=value,
        ),
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        0,
        1,
        "descending",
    ],
)
def test_plotting_rejects_invalid_axis_descending(
    operation,
    value,
):
    with pytest.raises(
        TypeError,
        match="axis_descending must be True, False, or None",
    ):
        operation(value)


@pytest.mark.parametrize(
    "spectra",
    [
        None,
        1,
        "not-a-sequence-of-spectra",
        (
            spectrum
            for spectrum in [_spectrum()]
        ),
    ],
)
def test_plot_spectra_rejects_nonsequence_input(
    spectra,
):
    with pytest.raises(
        TypeError,
        match="spectra must be a non-empty sequence",
    ):
        plot_spectra(spectra)


def test_plot_spectra_rejects_empty_or_malformed_sequence():
    with pytest.raises(
        ValueError,
        match="at least one Spectrum",
    ):
        plot_spectra([])

    with pytest.raises(
        TypeError,
        match="every item in spectra must be a Spectrum",
    ):
        plot_spectra(
            [
                _spectrum(),
                "not-a-spectrum",
            ]
        )


@pytest.mark.parametrize(
    "labels",
    [
        "single-string",
        1,
        {
            "first": "label",
        },
    ],
)
def test_plot_spectra_rejects_invalid_labels_container(
    labels,
):
    with pytest.raises(
        TypeError,
        match="labels must be a sequence of strings or None",
    ):
        plot_spectra(
            [_spectrum()],
            labels=labels,
        )


def test_plot_spectra_rejects_invalid_labels_contents():
    with pytest.raises(
        ValueError,
        match="one entry for each Spectrum",
    ):
        plot_spectra(
            [
                _spectrum(),
                _spectrum(),
            ],
            labels=["only-one"],
        )

    with pytest.raises(
        TypeError,
        match="every item in labels must be a string",
    ):
        plot_spectra(
            [
                _spectrum(),
                _spectrum(),
            ],
            labels=[
                "first",
                2,
            ],
        )


def test_plotting_does_not_use_structural_validation_as_gate():
    spectrum = _spectrum(
        axis=(3000.0, 2000.0, 2000.0, 2500.0),
        response=(0.10, np.nan, 0.30, 0.40),
        axis_unit="cm-1",
        response_type="Reflectance",
        response_unit="arbitrary",
    )

    ax = plot_spectrum(spectrum)

    np.testing.assert_array_equal(
        ax.lines[0].get_xdata(),
        spectrum.axis,
    )
    np.testing.assert_array_equal(
        ax.lines[0].get_ydata(),
        spectrum.response,
    )
    assert ax.get_xlabel() == "Axis"
    assert ax.get_ylabel() == "Response"


def test_plotting_does_not_mutate_source_state():
    spectra = [
        _spectrum(
            axis=(3000.0, 1000.0, 2000.0),
            response=(0.30, 0.10, 0.20),
            axis_unit="cm^-1",
            response_type="absorbance",
            spectrum_id="first",
            hierarchy={
                "source": {
                    "id": "source-a",
                },
                "replicate": 1,
            },
            metadata={
                "provenance": {
                    "import": {
                        "importer": "example",
                    },
                    "transformations": [
                        {
                            "operation": "example.operation",
                        },
                    ],
                }
            },
        ),
        _spectrum(
            axis=(3900.0, 2500.0),
            response=(0.15, 0.25),
            spectrum_id="second",
            hierarchy={
                "replicate": 2,
            },
            metadata={
                "notes": {
                    "operator": "analyst-a",
                }
            },
        ),
    ]
    before = [
        {
            "axis": spectrum.axis.copy(),
            "response": spectrum.response.copy(),
            "axis_dtype": spectrum.axis.dtype,
            "response_dtype": spectrum.response.dtype,
            "axis_unit": spectrum.axis_unit,
            "response_type": spectrum.response_type,
            "response_unit": spectrum.response_unit,
            "spectrum_id": spectrum.spectrum_id,
            "hierarchy": deepcopy(
                spectrum.hierarchy
            ),
            "metadata": deepcopy(
                spectrum.metadata
            ),
        }
        for spectrum in spectra
    ]

    plot_spectrum(spectra[0])
    plot_spectra(spectra)

    for spectrum, original in zip(
        spectra,
        before,
    ):
        np.testing.assert_array_equal(
            spectrum.axis,
            original["axis"],
        )
        np.testing.assert_array_equal(
            spectrum.response,
            original["response"],
        )
        assert (
            spectrum.axis.dtype
            == original["axis_dtype"]
        )
        assert (
            spectrum.response.dtype
            == original["response_dtype"]
        )
        assert (
            spectrum.axis_unit
            == original["axis_unit"]
        )
        assert (
            spectrum.response_type
            == original["response_type"]
        )
        assert (
            spectrum.response_unit
            == original["response_unit"]
        )
        assert (
            spectrum.spectrum_id
            == original["spectrum_id"]
        )
        assert (
            spectrum.hierarchy
            == original["hierarchy"]
        )
        assert (
            spectrum.metadata
            == original["metadata"]
        )
        assert not spectrum.axis.flags.writeable
        assert not spectrum.response.flags.writeable


def test_plotting_does_not_show_or_save_implicitly(
    monkeypatch,
):
    def fail(*args, **kwargs):
        raise AssertionError(
            "plotting helper attempted implicit output"
        )

    monkeypatch.setattr(
        plt,
        "show",
        fail,
    )
    monkeypatch.setattr(
        Figure,
        "show",
        fail,
    )
    monkeypatch.setattr(
        plt,
        "savefig",
        fail,
    )
    monkeypatch.setattr(
        Figure,
        "savefig",
        fail,
    )

    plot_spectrum(
        _spectrum()
    )
    plot_spectra(
        [
            _spectrum(),
            _spectrum(),
        ]
    )