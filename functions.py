import numpy as np
from PIL import Image, ImageDraw
from scipy.linalg import hadamard
def lightcheck(mat, ad): #ad - адамара, adC - изменённая
    t = mat.dot(ad)
    return not(np.linalg.det(t) == 0)
add = hadamard(8)
def blockmap(img):
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
            c=0
            if lightcheck(matr, add):
                c=255
                s1+=1
            if lightcheck(matg, add):
                c=255
                s2 += 1
            if lightcheck(matb, add):
                c=255
                s3 += 1
            t1 += 1
            t2 += 1
            t3 += 1
            for i in range(8):
                for j in range(8):
                    draw.point((ic + i, jc + j), (c, c, c))
            jc += 8
        jc = 0
        ic += 8
    new.save('2.bmp')
    print(s1/t1)
    print(s2/t2)
    print(s3/t3)