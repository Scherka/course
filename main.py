from PyQt5.QtWidgets import (
QApplication, QComboBox, QDialog, QWidget,
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

k1 = None
k2 = None
redDct = None
greenDct = None
blueDct = None
alphaMat = None
img = None
pic_split = []
def showImage(path, wid):
    wid.hide()
    pixmapimage = QPixmap(path)
    w, h = wid.width(), wid.height()
    pixmapimage = pixmapimage.scaled(w, h, Qt.KeepAspectRatio)
    wid.setPixmap(pixmapimage)
    wid.show()

def chooseCon():
    disableInject()
    disableExtract()
    global img, current_picture, pic_split, redDct, greenDct, blueDct, alphaMat, k1, k2
    file_dialog = QFileDialog()
    file_dialog.setFileMode(QFileDialog.ExistingFile)
    if file_dialog.exec_():
        pic_split = []
        current_picture = file_dialog.selectedFiles()[0]
        pic_split.append(os.path.splitext(current_picture)[0])
        pic_split.append(os.path.splitext(current_picture)[1])
        img = Image.open(current_picture)
        redDct, greenDct, blueDct, alphaMat = ko.readImage(img, func)
        result, k1_, k2_ = ko.getKs(redDct, greenDct, blueDct)
        k1 = k1_
        k2 = k2_
        # max_mes_len.setText(f"Максимальная длина сообщения в битах: {result}")
        pic_name.setText(
            f"Текущее изображение: {current_picture}\n"
            f"Максимальная длина сообщения в битах: {result}"
        )
        showImage(current_picture, lb_image)

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
    eps_value = eps_input.text().strip()
    
    try:
        eps_value = float(eps_value)
        ko.eps = eps_value
    except ValueError:
        # не число
        print('Используется eps по умолчанию')
        return
# def checkExtract():
#     try:
#         size_text = int(size_mes.text())
#         k1 = list(map(int, k1_mes.text().split()))
#         k2 = list(map(int, k2_mes.text().split()))
#         if all(0 <= num <= 7 for num in k1) and all(0 <= num <= 7 for num in k2) and len(k1) == 2 and len(k2) == 2 \
#                 and current_picture != "":
#             btn_extract.setEnabled(True)
#     except:
#         pass
def checkExtract():
    try:
        key = extract_key.text().strip()

        if not key.isdigit() or len(key) < 4 or current_picture == "":
            return

        # первые 2 цифры
        k1 = list(map(int, key[:2]))
        # вторые 2 цифры
        k2 = list(map(int, key[2:4]))
        # оставшиеся цифры – размер
        size_value = int(key[4:]) if len(key) > 4 else 0

        if (all(0 <= v <= 7 for v in k1) and
            all(0 <= v <= 7 for v in k2) and
            size_value >= 0):
            btn_extract.setEnabled(True)

    except:
        pass

def disableInject():
    btn_inject.setEnabled(False)

def disableExtract():
    btn_extract.setEnabled(False)

def injectMessage():
    # global img
    # img = Image.open(current_picture)
    # redDct, greenDct, blueDct, alphaMat = ko.readImage(img, func)
    result, k1, k2 = ko.getKs(redDct, greenDct, blueDct)
    key = f'{k1[0]}{k1[1]}{k2[0]}{k2[1]}{len(code(text_mes.text()))}'
    show_message_window("Ключ", key)
    print(key)
    print(code(text_mes.text()))
    injection_mode_text = 'lsb' if injection_mode.currentText() == "Встраивание по спирали" else 'linear'
    # injection_mode_text = 'linear'
    imgInjected = ko.inject(img, redDct, greenDct, blueDct, alphaMat, code(text_mes.text()), k1, k2, func, funcReverse, injection_mode_text)
    imgInjected.save( fr"{pic_split[0]}-injected_{ko.eps}{pic_split[1]}")
    # ko.writeImage(img, redDctRev, greenDctRev, blueDctRev, fr"{pic_split[0]}-injected{pic_split[1]}", funcReverse)

def extractMessage():
    global img
    global current_picture
    img = Image.open(current_picture)

    redDct, greenDct, blueDct, alphaMat = ko.readImage(img, func)
    k1, k2, s = split_digits(extract_key.text())
    print(k1, k2, s)
    injection_mode_text = 'lsb' if extract_mode.currentText() == "Изъятие по спирали" else 'linear'
    # injection_mode_text = 'linear'
    bin = ko.extract(img, redDct, greenDct, blueDct, k1, k2, int(s), func, injection_mode_text)
    if bin == None:
        print('Не удалось изъять сообщение')
    else:
        print(bin)
        print(decode(bin))
        show_message_window("Изъятое сообщение", f"{decode(bin)}\n{bin}")
def split_digits(s: str):
    s = s.strip()
    if not s.isdigit():
        raise ValueError("В строке должны быть только цифры")

    first = list(map(int, s[:2]))
    second = list(map(int, s[2:4]))
    rest = int(s[4:]) if len(s) > 4 else 0

    return first, second, rest

def show_message_window(title: str, text: str):
    # статический список внутри функции — окно не удалится GC
    if not hasattr(show_message_window, "_windows"):
        show_message_window._windows = []

    dlg = QDialog()
    dlg.setWindowTitle(title)
    dlg.setWindowFlags(Qt.Tool | Qt.WindowStaysOnTopHint)  # уведомление поверх
    dlg.setAttribute(Qt.WA_DeleteOnClose, False)            # не удалять объект
    dlg.setMinimumSize(200, 50)

    layout = QVBoxLayout(dlg)
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    layout.addWidget(label)

    dlg.show()

    show_message_window._windows.append(dlg)  # сохраняем для предотвращения исчезновения
    return dlg
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
eps_input = QLineEdit()
eps_input.setPlaceholderText("Введите коэффициент встраивания")
eps_input.setText('5')
injection_mode = QComboBox()
injection_mode.addItem("Встраивание по спирали")
injection_mode.addItem("Встраивание с края")
lb_full = QLabel("Заполненный контейнер")
btn_con = QPushButton("Выбрать изображение")
pic_name = QLabel("Текущее изображение: \nМаксимальная длина сообщения в битах: ")
# max_mes_len = QLabel("Максимальная длина сообщения в битах: ")
btn_inject = QPushButton("Вставить сообщение")
btn_inject.setEnabled(False)
btn_extract = QPushButton("Изъять сообщение")
btn_extract.setEnabled(False)
btn_check_mes = QPushButton("Проверить сообщение")
btn_check_extract = QPushButton("Проверить параметры изъятия")
extract_key = QLineEdit()
extract_key.setPlaceholderText("Введите ключ для изъятия сообщения сообщения")
extract_mode = QComboBox()
extract_mode.addItem("Изъятие по спирали")
extract_mode.addItem("Изъятие с края")
# k1_mes = QLineEdit()
# k1_mes.setPlaceholderText("Введите координаты первого коэффициента через пробел")
# k2_mes = QLineEdit()
# k2_mes.setPlaceholderText("Введите координаты второго коэффициента через пробел")

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
# colRow1.addWidget(max_mes_len, alignment=Qt.AlignTop | Qt.AlignHCenter)
row1.addLayout(colRow1)
row2.addWidget(lb_image, alignment=Qt.AlignCenter)
row3.addWidget(text_mes)
row3.addWidget(eps_input)
row3.addWidget(injection_mode)
row3.addWidget(btn_check_mes)
row4.addWidget(btn_inject)
row5.addWidget(extract_key)
row5.addWidget(extract_mode)
# row5.addWidget(k1_mes)
# row5.addWidget(k2_mes)
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

extract_key.textChanged.connect(disableExtract)
eps_input.textChanged.connect(disableInject)
# k1_mes.textChanged.connect(disableExtract)
# k2_mes.textChanged.connect(disableExtract)
win.show()
app.exec()