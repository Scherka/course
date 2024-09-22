from PIL import Image, ImageDraw
import numpy as np
from decimal import Decimal

eps = 5  # сила встраивания
# координаты внутри блоков, соответствующие средней частотной полосе
coefs = [[0, 6], [0, 7], [1, 6], [1, 7], [2, 4], [2, 5], [3, 4], [3, 5], [4, 2], [4, 3], [5, 2], [5, 3], [6, 0], [6, 1],
         [7, 0], [7, 1]]
redMat = []
redMatOg = []
greenMat = []
blueMat = []


# проверка матрицы одного цвета на возможность встраивания бита 1
def checkColor1(mat, k1, k2):
    # print(mat[k1[0]][k1[1]], mat[k2[0]][k2[1]])
    return abs(mat[k1[0]][k1[1]]) - abs(mat[k2[0]][k2[1]]) <= eps


# проверка матрицы одного цвета на возможность встраивания бита 0
def checkColor0(mat, k1, k2):
    return (abs(mat[k2[0]][k2[1]]) - abs(mat[k1[0]][k1[1]])) <= eps


def checkAll0(redMat, greenMat, blueMat, k1, k2):
    return not (checkColor0(redMat, k1, k2) and checkColor0(greenMat, k1, k2) and checkColor0(blueMat, k1, k2))


def checkAll1(redMat, greenMat, blueMat, k1, k2):
    return not (checkColor1(redMat, k1, k2) and checkColor1(greenMat, k1, k2) and checkColor1(blueMat, k1, k2))


# проход по матрицам частотных коэффициентов
def getKs():
    countMax = 0
    maxk1 = 0
    maxk2 = 0
    for i in range(len(coefs)):
        for j in range(len(coefs)):
            countSuccess = 0
            flag = True
            if i != j:
                for k in range(len(redMat)):
                    if k % 4 == 0:
                        if checkAll0(redMat[k], greenMat[k], blueMat[k], coefs[i], coefs[j]) \
                                and checkAll1(redMat[k], greenMat[k], blueMat[k], coefs[i], coefs[j]):
                            flag = False
                        else:
                            countSuccess += 1
                # if flag:
                #    return coefs[i], coefs[j]
            # print(countSuccess)
            if countSuccess > countMax:
                maxk1 = coefs[i]
                maxk2 = coefs[j]
                countMax = countSuccess
    return countMax, maxk1, maxk2


def inject(mes, k1, k2):
    i = 0
    for k in range(len(mes)):
        if mes[k] == '0':
            redMat[i] = injectMat0(redMat[k], k1, k2)
        if mes[k] == '1':
            redMat[i] = injectMat1(redMat[k], k1, k2)
        i += 4


def inject1(k1, k2):
    for k in range(len(redMat)):
        if (k - 1) % 4 == 0:
            redMat[k] = injectMat1(redMat[k], k1, k2)


def injectMat1(matOrig, k1, k2):
    # print(matOrig)
    mat = matOrig.copy()
    if mat[k1[0]][k1[1]] >= 0:
        mat[k1[0]][k1[1]] = abs(mat[k2[0]][k2[1]]) + eps
    else:
        mat[k1[0]][k1[1]] = -abs(mat[k2[0]][k2[1]]) - eps
    # print(mat)
    return mat


def inject0(k1, k2):
    for k in range(len(redMat)):
        if k % 4 == 0:
            redMat[k] = injectMat0(redMat[k], k1, k2)


def injectMat0(matOrig, k1, k2):
    # print(matOrig)
    mat = matOrig.copy()
    if mat[k2[0]][k2[1]] >= 0:
        mat[k2[0]][k2[1]] = abs(mat[k1[0]][k1[1]]) + eps
    else:
        mat[k2[0]][k2[1]] = -abs(mat[k1[0]][k1[1]]) - eps
    # print(mat)
    return mat


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


counter = 0
watermark = '01'
img = Image.open(r".\Pictures\Girl.bmp")
wc, hc = img.size  # получение размеров путсого контейнер
pix = img.load()
new = Image.new("RGB", (wc, hc))
draw = ImageDraw.Draw(new)
i, j, ic, jc = 0, 0, 0, 0
t1 = t2 = t3 = 0
s1 = s2 = s3 = 0
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
            redMatOg.append(matr)
            redMat.append(dctMul(matr))
            greenMat.append(dctMul(matg))
            blueMat.append(dctMul(matb))
            counter += 1
        jc += 8
    jc = 0
    ic += 8
# print(len(redMat))
result, k1, k2 = getKs()
print(result, k1, k2)

inject(watermark, k1, k2)

''''
inject0(k1, k2)
redMatRev = dctMulReverse(redMat[0])
getRedBit = dctMul(redMatRev)
print(abs(getRedBit[k1[0]][k1[1]]) > abs(getRedBit[k2[0]][k2[1]]))
inject1(k1, k2)
redMatRev = dctMulReverse(redMat[1])
print(redMatOg[1])
print(redMatRev)
getRedBit = dctMul(redMatRev)
print(abs(getRedBit[k1[0]][k1[1]]) > abs(getRedBit[k2[0]][k2[1]]))
'''
'''

result, k1, k2 = getKs1()
print(result, k1, k2)
'''
'''''

'''
'''

#print(mat1-redMat[0])
#print(redMatOg[0])

#print(redMatRev[k1[0]][k1[1]])

#print(getRedBit)
#print(abs(getRedBit[k1[0]][k1[1]]), abs(getRedBit[k2[0]][k2[1]]))
#
'''
