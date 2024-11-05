from PIL import Image, ImageDraw
import numpy as np
from scipy.linalg import hadamard
eps = 5
coefs = [[0, 6], [0, 7], [1, 6], [1, 7], [2, 4], [2, 5], [3, 4], [3, 5], [4, 2], [4, 3], [5, 2], [5, 3], [6, 0], [6, 1],
         [7, 0], [7, 1]]
hadamard_matrix = hadamard(8)

def checkColor1(mat, k1, k2):
    return abs(mat[k1[0]][k1[1]]) - abs(mat[k2[0]][k2[1]]) <= eps

def checkColor0(mat, k1, k2):
    return (abs(mat[k2[0]][k2[1]]) - abs(mat[k1[0]][k1[1]])) <= eps

def checkAll0(redDct, greenDct, blueDct, k1, k2):
    return not (checkColor0(redDct, k1, k2) and checkColor0(greenDct, k1, k2) and checkColor0(blueDct, k1, k2))

def checkAll1(redDct, greenDct, blueDct, k1, k2):
    return not (checkColor1(redDct, k1, k2) and checkColor1(greenDct, k1, k2) and checkColor1(blueDct, k1, k2))

def getKs(redDct, greenDct, blueDct):
    countMax = 0
    maxk1 = 0
    maxk2 = 0
    for i in range(len(coefs)):
        for j in range(len(coefs)):
            countSuccess = 0
            flag = True
            if i != j:
                for k in range(len(redDct)):
                    if k % 1 == 0:
                        if checkAll0(redDct[k], greenDct[k], blueDct[k], coefs[i], coefs[j]) \
                                and checkAll1(redDct[k], greenDct[k], blueDct[k], coefs[i], coefs[j]):
                            flag = False
                        else:
                            countSuccess += 1
            if countSuccess > countMax:
                maxk1 = coefs[i]
                maxk2 = coefs[j]
                countMax = countSuccess
    return countMax, maxk1, maxk2

def inject(redDct, greenDct, blueDct, mes, k1, k2):
    i = 0
    for k in range(len(mes)):
        if mes[k] == '0':
            redDct[i] = injectMat0(redDct[i], k1, k2)
            greenDct[i] = injectMat0(greenDct[i], k1, k2)
            blueDct[i] = injectMat0(blueDct[i], k1, k2)
        if mes[k] == '1':
            redDct[i] = injectMat1(redDct[i], k1, k2)
            greenDct[i] = injectMat1(greenDct[i], k1, k2)
            blueDct[i] = injectMat1(blueDct[i], k1, k2)
        i += 4
    return redDct, greenDct, blueDct

def injectMat1(matOrig, k1, k2):
    mat = matOrig.copy()
    if mat[k1[0]][k1[1]] >= 0:
        mat[k1[0]][k1[1]] = abs(mat[k2[0]][k2[1]]) + eps
    else:
        mat[k1[0]][k1[1]] = -abs(mat[k2[0]][k2[1]]) - eps
    return mat

def injectMat0(matOrig, k1, k2):
    mat = matOrig.copy()
    if mat[k2[0]][k2[1]] >= 0:
        mat[k2[0]][k2[1]] = abs(mat[k1[0]][k1[1]]) + eps
    else:
        mat[k2[0]][k2[1]] = -abs(mat[k1[0]][k1[1]]) - eps
    return mat

def extract(redDct, greenDct, blueDct, k1, k2, l):
    mes = ""
    for k in range(l):
        if k%4==0:
            c_bit = 0
            if abs(redDct[k][k1[0]][k1[1]]) > abs(redDct[k][k2[0]][k2[1]]): c_bit+=1
            if abs(greenDct[k][k1[0]][k1[1]]) > abs(greenDct[k][k2[0]][k2[1]]): c_bit += 1
            if abs(blueDct[k][k1[0]][k1[1]]) > abs(blueDct[k][k2[0]][k2[1]]): c_bit += 1
            if c_bit>=2:
                mes += '1'
            else:
                mes += '0'
    return mes


def dctCreate():
    dctMat = []
    for i in range(8):
        dctMat.append([])
        for j in range(8):
            if i == 0:
                dctMat[i].append(np.sqrt((1 / 8)))
            else:
                dctMat[i].append(np.sqrt((1 / 4)) * np.cos(((2 * j + 1) * i * np.pi) / (16)))
    return np.array(dctMat)

dct = dctCreate()
def dctMul(mat):
    mat1 = np.matmul(np.transpose(dct), mat)
    return np.matmul(mat1, dct)
def dctMulReverse(mat):
    mat1 = np.matmul(dct, mat)
    return np.matmul(mat1, np.transpose(dct)).astype(int)
def hadMul(mat):
    mat1 = np.matmul(np.transpose(hadamard_matrix), mat)
    return np.matmul(mat1, hadamard_matrix)//8
def hadMulReverse(mat):
    mat1 = np.matmul(hadamard_matrix, mat)
    return np.matmul(mat1, np.transpose(hadamard_matrix))//8

def readImage(img, func):
    redDct = []
    greenDct = []
    blueDct = []
    counter = 0
    wc, hc = img.size
    pix = img.load()
    i, j, ic, jc = 0, 0, 0, 0
    while ic + 8 <= wc:
        while jc + 8 <= hc:
            mat1r = []
            mat1g = []
            mat1b = []
            for i in range(ic, ic + 8):
                mat2r = []
                mat2g = []
                mat2b = []
                for j in range(jc, jc + 8):
                    c0, c1, c2 = pix[i, j]
                    mat2r.append(c0)
                    mat2g.append(c1)
                    mat2b.append(c2)
                mat1r.append(mat2r)
                mat1g.append(mat2g)
                mat1b.append(mat2b)
            matr = np.array(mat1r)
            matg = np.array(mat1g)
            matb = np.array(mat1b)
            if counter < 8:
                redDct.append(func(matr))
                greenDct.append(func(matg))
                blueDct.append(func(matb))
            jc += 8
        jc = 0
        ic += 8
    return redDct, greenDct, blueDct

def writeImage(img, redDct, greenDct, blueDct, newFile, func):
    wc, hc = img.size
    new = Image.new("RGB", (wc, hc))
    draw = ImageDraw.Draw(new)
    i, j, ic, jc = 0, 0, 0, 0
    k = 0
    redMat =[]
    greenMat = []
    blueMat = []
    while ic + 8 <= wc:
        while jc + 8 <= hc:
            redMat.append(func(redDct[k]))
            greenMat.append(func(greenDct[k]))
            blueMat.append(func(blueDct[k]))
            for i in range(8):
                for j in range(8):
                    draw.point((ic + i, jc + j), (redMat[k][i][j], greenMat[k][i][j], blueMat[k][i][j]))
            k+=1
            jc += 8
        jc = 0
        ic += 8
    new.save(newFile)