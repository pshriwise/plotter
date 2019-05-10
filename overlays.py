
from PySide2 import QtGui, QtCore
from PySide2.QtWidgets import (QWidget, QTableWidget, QSizePolicy,
                               QPushButton, QTableWidgetItem, QVBoxLayout)


class ShortcutsOverlay(QWidget):
    shortcuts = {"Color By": [("Cell", "Alt + C"),
                              ("Material", "Alt + M"),
                              ("Temperature", "Alt + T"),
                              ("Density", "Alt + D")],
                 "View Options": [("Apply Changes", "Ctrl + Enter"),
                                  ("Undo", "Ctrl + Z"),
                                  ("Redo", "Shift + Ctrl + Z"),
                                  ("Restore Default Plot", "Ctrl + R"),
                                  ("Zoom", "Alt + Shift + Z"),
                                  ("Zoom", "Shift + scroll"),
                                  ("Toggle Masking", "Ctrl + M"),
                                  ("Toggle Highlighting", "Ctrl + L"),
                                  ("Set XY Basis", "Alt + X"),
                                  ("Set YZ Basis", "Alt + Y"),
                                  ("Set XZ Basis", "Alt + Z"),
                                  ("Update Plot Origin", "double-click"),
                                  ("Open Context Menu", "right-click")],
                 "Menus": [("Hide/Show Options Dock", "Ctrl + D"),
                           ("Save View", "Ctrl + S"),
                           ("Open View", "Ctrl + O"),
                           ("Save Plot Image", "Ctrl + Shift + S"),
                           ("Quit", "Ctrl + D"),
                           ("Display Shortcuts", "?")]}

    # colors
    header_color = QtGui.QColor(150, 150, 150, 255)
    fillColor = QtGui.QColor(30, 30, 30, 200)
    framePenColor = QtGui.QColor(255, 255, 255, 120)
    textPenColor = QtGui.QColor(152, 196, 5, 255)

    def __init__(self, parent):
        super().__init__(parent)

        # transparent window fill
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        n_rows = max((len(scts) for scts in self.shortcuts.values()))
        n_rows += 1  # plus one for header
        n_cols = len(self.shortcuts.keys()) * 3
        self.tableWidget = QTableWidget(n_rows, n_cols, self)
        self.layout.addWidget(self.tableWidget)

        self.tableWidget.setShowGrid(False)
        self.tableWidget.setSizePolicy(QSizePolicy.Expanding,
                                       QSizePolicy.Expanding)
        self.tableWidget.verticalHeader().setVisible(False)
        self.tableWidget.horizontalHeader().setVisible(False)
        self.tableWidget.setStyleSheet("background-color:"
                                       "rgba(30, 30, 30, 230);"
                                       "border: 0px;"
                                       "padding: 20px")

        # populate table cells
        self.set_cells()

        self.close_btn = QPushButton(self)
        self.close_btn.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.close_btn.setStyleSheet("background-color: rgba(0, 0, 0, 0);"
                                     "border: 0px;"
                                     "color: rgba(150, 150, 150, 255)")
        self.close_btn.setText("X")
        font = QtGui.QFont()
        self.close_btn.setFixedSize(30, 30)
        self.close_btn.clicked.connect(self.hide)

    def set_cells(self):
        # row, col indices
        row_idx = 0
        col_idx = 0

        for menu in self.shortcuts:
            # set menu header
            header_item = QTableWidgetItem()
            header_item.setTextColor(QtGui.QColor(150, 150, 150, 255))
            header_item.setText(menu)
            self.tableWidget.setItem(row_idx, col_idx, header_item)
            row_idx += 1

            for shortcut in self.shortcuts[menu]:
                desc_item = QTableWidgetItem()
                desc_item.setTextColor(self.textPenColor)
                desc_item.setText(shortcut[0])
                self.tableWidget.setItem(row_idx, col_idx, desc_item)

                key_item = QTableWidgetItem()
                key_item.setTextColor(self.textPenColor)
                key_item.setText(shortcut[1])
                self.tableWidget.setItem(row_idx, col_idx + 1, key_item)
                row_idx += 1
            # update for next menu
            row_idx = 0
            col_idx += 3

        self.tableWidget.resizeColumnsToContents()

    def keyPressEvent(self, event):
        # close if visible and esc is pressed
        if self.isVisible() and event.key() == QtCore.Qt.Key_Escape:
            self.close()

    def resizeEvent(self, event):
        overlay_size = self.size()
        btn_size = self.close_btn.size()
        x_pos = int(overlay_size.width() - btn_size.width()) - 5
        self.close_btn.move(x_pos, 5)
