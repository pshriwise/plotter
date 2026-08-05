from types import SimpleNamespace

import numpy as np
import pytest
from PySide6 import QtGui, QtWidgets

from openmc_plotter.plotgui import PlotImage

class FakeClipboard:

    def __init__(self):
        self.image = None

    def setImage(self, image):
        self.image = image


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_copy_image_to_clipboard_crops_to_plot_and_preserves_alpha(
    qapp, monkeypatch
):
    scroll = QtWidgets.QScrollArea()
    scroll.resize(220, 160)
    main_window = SimpleNamespace(
        logicalDpiX=lambda: 100,
        zoom=100,
        coord_label=SimpleNamespace(show=lambda: None, hide=lambda: None),
        statusBar=lambda: SimpleNamespace(showMessage=lambda *args, **kwargs: None),
    )
    plot = PlotImage(model=None, parent=scroll, main_window=main_window)
    scroll.setWidget(plot)
    plot.resize(400, 300)
    plot.figure.clear()
    plot.ax = plot.figure.subplots()
    plot.ax.imshow(np.zeros((10, 10, 4)))
    scroll.show()
    qapp.processEvents()
    scroll.horizontalScrollBar().setValue(40)
    scroll.verticalScrollBar().setValue(30)
    qapp.processEvents()

    fake_clipboard = FakeClipboard()
    monkeypatch.setattr(
        QtGui.QGuiApplication,
        "clipboard",
        staticmethod(lambda: fake_clipboard),
    )

    try:
        assert plot.copyImageToClipboard()
    finally:
        plot.close()
        scroll.close()

    assert fake_clipboard.image is not None
    assert not fake_clipboard.image.isNull()
    assert fake_clipboard.image.hasAlphaChannel()

    canvas_width, canvas_height = plot.get_width_height()
    expected_width = round(
        min(scroll.viewport().width(), plot.width()) * canvas_width / plot.width()
    )
    expected_height = round(
        min(scroll.viewport().height(), plot.height()) * canvas_height / plot.height()
    )
    assert fake_clipboard.image.width() == pytest.approx(expected_width, abs=1)
    assert fake_clipboard.image.height() == pytest.approx(expected_height, abs=1)
    assert fake_clipboard.image.width() < canvas_width
    assert fake_clipboard.image.height() < canvas_height

    center = fake_clipboard.image.pixelColor(
        fake_clipboard.image.width() // 2,
        fake_clipboard.image.height() // 2,
    )
    assert center.alpha() == 0


def test_surface_crossing_contours_use_surface_id_labels(qapp, monkeypatch):
    scroll = QtWidgets.QScrollArea()
    main_window = SimpleNamespace(
        logicalDpiX=lambda: 100,
        zoom=100,
        xBasis=0,
        yBasis=1,
        coord_label=SimpleNamespace(show=lambda: None, hide=lambda: None),
        statusBar=lambda: SimpleNamespace(showMessage=lambda *args, **kwargs: None),
    )
    current_view = SimpleNamespace(
        useRaytracedPlots=True,
        showSurfaceIDs=True,
        outlinesCell=False,
        surface_crossing_color=(0, 0, 0),
        origin=(0.0, 0.0, 0.0),
        width=4.0,
        height=4.0,
    )
    cell_ids = np.array([
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [3, 3, 4, 4],
        [3, 3, 4, 4],
    ], dtype=np.int32)
    surface_crossing_map = np.ma.masked_equal(np.array([
        [11, 11, -1, 7],
        [-1, 11, 7, 7],
        [-1, 11, 7, -1],
        [-1, -1, 7, -1],
    ], dtype=float), -1)
    surface_crossing_ids = {7, 11}
    model = SimpleNamespace(
        currentView=current_view,
        cell_ids=cell_ids,
        surface_crossing_map=surface_crossing_map,
        surface_crossing_ids=surface_crossing_ids,
    )

    plot = PlotImage(model=model, parent=scroll, main_window=main_window)
    scroll.setWidget(plot)
    plot.figure.clear()
    plot.ax = plot.figure.subplots()

    contour_calls = []
    text_calls = []
    original_contour = plot.ax.contour
    original_text = plot.ax.text

    def record_contour(*args, **kwargs):
        contour_calls.append({
            'data': np.asarray(args[0]),
            **kwargs,
        })
        return original_contour(*args, **kwargs)

    def record_text(*args, **kwargs):
        text_calls.append({
            'x': args[0],
            'y': args[1],
            'text': args[2],
            **kwargs,
        })
        return original_text(*args, **kwargs)

    monkeypatch.setattr(plot.ax, 'contour', record_contour)
    monkeypatch.setattr(plot.ax, 'text', record_text)

    try:
        existing_collections = len(plot.ax.collections)
        plot.add_surface_crossing_contours()
    finally:
        plot.close()
        scroll.close()

    assert len(plot.ax.collections) > existing_collections
    assert len(contour_calls) == 1
    assert np.array_equal(contour_calls[0]['data'], cell_ids)
    assert contour_calls[0]['linewidths'] == 2.5
    assert contour_calls[0]['zorder'] == 10
    assert set(contour_calls[0]['levels']) == set(np.unique(cell_ids))
    assert len(text_calls) == len(surface_crossing_ids)
    assert [call['text'] for call in text_calls] == [
        f'Surface {surface_id}' for surface_id in sorted(surface_crossing_ids)
    ]
    assert all(call['zorder'] == 11 for call in text_calls)


def test_surface_crossing_contours_can_hide_surface_id_labels(qapp, monkeypatch):
    scroll = QtWidgets.QScrollArea()
    main_window = SimpleNamespace(
        logicalDpiX=lambda: 100,
        zoom=100,
        xBasis=0,
        yBasis=1,
        coord_label=SimpleNamespace(show=lambda: None, hide=lambda: None),
        statusBar=lambda: SimpleNamespace(showMessage=lambda *args, **kwargs: None),
    )
    current_view = SimpleNamespace(
        useRaytracedPlots=True,
        showSurfaceIDs=False,
        outlinesCell=False,
        surface_crossing_color=(0, 0, 0),
        origin=(0.0, 0.0, 0.0),
        width=4.0,
        height=4.0,
    )
    cell_ids = np.array([
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [3, 3, 4, 4],
        [3, 3, 4, 4],
    ], dtype=np.int32)
    surface_crossing_map = np.ma.masked_equal(np.array([
        [11, 11, -1, 7],
        [-1, 11, 7, 7],
        [-1, 11, 7, -1],
        [-1, -1, 7, -1],
    ], dtype=float), -1)
    model = SimpleNamespace(
        currentView=current_view,
        cell_ids=cell_ids,
        surface_crossing_map=surface_crossing_map,
        surface_crossing_ids={7, 11},
    )

    plot = PlotImage(model=model, parent=scroll, main_window=main_window)
    scroll.setWidget(plot)
    plot.figure.clear()
    plot.ax = plot.figure.subplots()

    contour_calls = []
    text_calls = []
    original_contour = plot.ax.contour
    original_text = plot.ax.text

    def record_contour(*args, **kwargs):
        contour_calls.append(kwargs)
        return original_contour(*args, **kwargs)

    def record_text(*args, **kwargs):
        text_calls.append((args, kwargs))
        return original_text(*args, **kwargs)

    monkeypatch.setattr(plot.ax, 'contour', record_contour)
    monkeypatch.setattr(plot.ax, 'text', record_text)

    try:
        plot.add_surface_crossing_contours()
    finally:
        plot.close()
        scroll.close()

    assert contour_calls == []
    assert text_calls == []
