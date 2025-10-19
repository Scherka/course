from PyQt5.QtWidgets import (
QApplication, QWidget,
QFileDialog,
QLabel, QPushButton,
QHBoxLayout, QVBoxLayout, QLineEdit
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PIL import Image

import koch as ko
from code import code, decode
import os

def showImage(path, wid):
    wid.hide()
    pixmapimage = QPixmap(path)
    w, h = wid.width(), wid.height()
    pixmapimage = pixmapimage.scaled(w, h, Qt.KeepAspectRatio)
    wid.setPixmap(pixmapimage)
    wid.show()

def chooseCon():
    global current_picture
    global pic_split
    pic_split = []
    file_dialog = QFileDialog()
    file_dialog.setFileMode(QFileDialog.ExistingFile)
    if file_dialog.exec_():
        current_picture = file_dialog.selectedFiles()[0]
        pic_split.append(os.path.splitext(current_picture)[0])
        pic_split.append(os.path.splitext(current_picture)[1])
        showImage(current_picture, lb_image)
    pic_name.setText(f"Текущее изображение: {current_picture}")

def chooseMes():
    global workdir2
    file_dialog = QFileDialog()
    file_dialog.setFileMode(QFileDialog.ExistingFile)
    file_dialog.exec_()
    workdir2 = file_dialog.selectedFiles()[0]
    showImage(workdir2, lb_mes)

def checkMessage():
    btn_inject.setEnabled(True)
    line_text = text_mes.text()
    if line_text != "" and line_text.replace("0", "").replace("1", "") == "" and current_picture != "":
        btn_inject.setEnabled(True)

def checkExtract():
    try:
        size_text = int(size_mes.text())
        k1 = list(map(int, k1_mes.text().split()))
        k2 = list(map(int, k2_mes.text().split()))
        if all(0 <= num <= 7 for num in k1) and all(0 <= num <= 7 for num in k2) and len(k1) == 2 and len(k2) == 2 \
                and current_picture != "":
            btn_extract.setEnabled(True)
    except:
        pass

def disableInject():
    btn_inject.setEnabled(False)

def disableExtract():
    btn_extract.setEnabled(False)

def injectMessage():
    ko.redDct = []
    ko.redDctOg = []
    ko.greenDct = []
    ko.blueDct = []
    redDct, greenDct, blueDct = ko.readImage(Image.open(current_picture), func)
    result, k1, k2 = ko.getKs(redDct, greenDct, blueDct)
    print(len(code(text_mes.text())), k1[0], k1[1], k2[0], k2[1])
    redDctRev, greenDctRev, blueDctRev = ko.inject(redDct, greenDct, blueDct, code(text_mes.text()), k1, k2)
    ko.writeImage(Image.open(current_picture), redDctRev, greenDctRev, blueDctRev, fr"{pic_split[0]}-injected{pic_split[1]}", funcReverse)

def extractMessage():
    redDct, greenDct, blueDct = ko.readImage(Image.open(fr"{current_picture}"), func)
    s = size_mes.text()
    k1 = list(map(int, k1_mes.text().split()))
    k2 = list(map(int, k2_mes.text().split()))
    bin = ko.extract(redDct, greenDct, blueDct, k1, k2, int(s)*4)
    print(decode(bin))

current_picture = ""
func = ko.hadMul
funcReverse = ko.hadMulReverse
app = QApplication([])
win = QWidget()
win.resize(1920, 1000)
win.setWindowTitle('Стеганограф')
map_image = QLabel("Карта")
lb_image = QLabel()
lb_image.setMinimumSize(700,700)
lb_mes = QLabel("Сообщение")
text_mes = QLineEdit()
text_mes.setPlaceholderText("Введите сообщение")
lb_full = QLabel("Заполненный контейнер")
btn_con = QPushButton("Выбрать изображение")
pic_name = QLabel("Текущее изображение: ")
btn_inject = QPushButton("Вставить сообщение")
btn_inject.setEnabled(False)
btn_extract = QPushButton("Изъять сообщение")
btn_extract.setEnabled(False)
btn_check_mes = QPushButton("Проверить сообщение")
btn_check_extract = QPushButton("Проверить параметры изъятия")
size_mes = QLineEdit()
size_mes.setPlaceholderText("Введите длину сообщения")
k1_mes = QLineEdit()
k1_mes.setPlaceholderText("Введите координаты первого коэффициента через пробел")
k2_mes = QLineEdit()
k2_mes.setPlaceholderText("Введите координаты второго коэффициента через пробел")

col = QVBoxLayout()
colRow1 = QVBoxLayout()
row1 = QHBoxLayout()
row2 = QHBoxLayout()
row3 = QHBoxLayout()
row4 = QHBoxLayout()
row5 = QHBoxLayout()
row6 = QHBoxLayout()

colRow1.setSpacing(0)
colRow1.setContentsMargins(0, 0, 0, 0)
colRow1.addWidget(btn_con)
colRow1.addWidget(pic_name, alignment=Qt.AlignTop | Qt.AlignHCenter)
row1.addLayout(colRow1)
row2.addWidget(lb_image, alignment=Qt.AlignCenter)
row3.addWidget(text_mes)
row3.addWidget(btn_check_mes)
row4.addWidget(btn_inject)
row5.addWidget(size_mes)
row5.addWidget(k1_mes)
row5.addWidget(k2_mes)
row5.addWidget(btn_check_extract)
row6.addWidget(btn_extract)
col.addLayout(row1)
col.addLayout(row2)
col.setAlignment(row2, Qt.AlignVCenter)
col.addLayout(row3)
col.addLayout(row4)
col.addLayout(row5)
col.addLayout(row6)
win.setLayout(col)

btn_con.clicked.connect(chooseCon)
btn_check_mes.clicked.connect(checkMessage)
btn_check_extract.clicked.connect(checkExtract)
text_mes.textChanged.connect(disableInject)
btn_inject.clicked.connect(injectMessage)
btn_extract.clicked.connect(extractMessage)

size_mes.textChanged.connect(disableExtract)
k1_mes.textChanged.connect(disableExtract)
k2_mes.textChanged.connect(disableExtract)
win.show()
app.exec()