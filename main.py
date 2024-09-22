from PyQt5.QtWidgets import (
QApplication, QWidget,
QFileDialog, # Диалог открытия файлов (и папок)
QLabel, QPushButton, QListWidget,
QHBoxLayout, QVBoxLayout
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PIL import Image, ImageFilter
import functions as fun
import os


def showImage(path, wid):
    wid.hide()
    pixmapimage = QPixmap(path)
    w, h = wid.width(), wid.height()
    pixmapimage = pixmapimage.scaled(w, h, Qt.KeepAspectRatio)
    wid.setPixmap(pixmapimage)
    wid.show()



def chooseCon():
    global workdir
    file_dialog = QFileDialog()
    file_dialog.setFileMode(QFileDialog.ExistingFile)
    file_dialog.exec_()
    workdir = file_dialog.selectedFiles()[0]
    con = Image.open(workdir)
    fun.blockmap(con)
    showImage(workdir, lb_image)
    showImage('2.bmp', map_image)

def chooseMes():
    global workdir2
    file_dialog = QFileDialog()
    file_dialog.setFileMode(QFileDialog.ExistingFile)
    file_dialog.exec_()
    workdir2 = file_dialog.selectedFiles()[0]
    con = Image.open(workdir2)
    showImage(workdir2, lb_mes)

workdir = ''
app = QApplication([])
win = QWidget()
win.resize(1920, 1000)
win.setWindowTitle('Стеганограф')
lb_image = QLabel("Картинка")
map_image = QLabel("Карта")
lb_mes = QLabel("Сообщение")
lb_full = QLabel("Заполненный контейнер")
btn_con = QPushButton("Контейнер")
btn_mes = QPushButton("Сообщение")
row = QVBoxLayout() # Основная строка
col1 = QHBoxLayout() # делится на два столбца
col2 = QHBoxLayout()
col3 = QHBoxLayout()
col4 = QHBoxLayout()
col1.addWidget(btn_con) # в первом - кнопка выбора директории
col2.addWidget(lb_image, 300)# вo втором - картинка
col2.addWidget(map_image, 300)
col3.addWidget(btn_mes,300)
col4.addWidget(lb_mes,300)
col4.addWidget(lb_full,300)
row.addLayout(col1)
row.addLayout(col2)
row.addLayout(col3)
row.addLayout(col4)
win.setLayout(row)


#con = Image.open(workimage.filename ) #закгрузка контейнера
btn_con.clicked.connect(chooseCon)
btn_mes.clicked.connect(chooseMes)
win.show()
app.exec()