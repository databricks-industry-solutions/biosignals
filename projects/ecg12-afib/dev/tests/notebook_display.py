"""Matplotlib backend for the tests: Agg, off-screen, with ``plt.show()`` drawing the open figures,
as a Databricks notebook renders them into its output, instead of warning that Agg cannot show
them (matplotlib warns only where a display could exist: not on Linux without ``DISPLAY``).
conftest.py selects it with ``MPLBACKEND=module://notebook_display``."""
from matplotlib.backend_bases import FigureManagerBase
from matplotlib.backends.backend_agg import FigureCanvasAgg


class FigureManager(FigureManagerBase):
    def show(self):
        self.canvas.draw()


class FigureCanvas(FigureCanvasAgg):
    manager_class = FigureManager
