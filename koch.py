import json
from PIL import Image, ImageDraw
import numpy as np
from scipy.linalg import hadamard
eps = 5
dct_marker = 75

marker_check_coords = [[7,0], [6,1], [0,7], [1,6]]
coefs = [[0, 6], [0, 7], [1, 6], [1, 7], [2, 4], [2, 5], [3, 4], [3, 5], [4, 2], [4, 3], [5, 2], [5, 3], [6, 0], [6, 1],
         [7, 0], [7, 1]]
ls_bit = 1
hadamard_matrix = hadamard(8)
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
mark_func = hadMul  
mark_func_reverse = hadMulReverse
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
def replace_block(i_start, j_start, redMatrix, greenMatrix, blueMatrix, alphaMat, pixels):
    block_size = 8
    for i in range(block_size):
        for j in range(block_size):
            if alphaMat[i, j] == None:
                pixels[i_start + i, j_start + j] = (redMatrix[i, j], greenMatrix[i, j], blueMatrix[i, j])
            else:
                pixels[i_start + i, j_start + j] = (redMatrix[i, j], greenMatrix[i, j], blueMatrix[i, j], alphaMat[i, j])
    return pixels


def mark_center(width, height, pixels, func, funcReverse):
    block_size = 8
    print('mark_center', width // 2 - 4, height // 2 - 4)

    r_arr, g_arr, b_arr, alpha_arr = [], [], [], []

    for i in range(width // 2 - 4, width // 2 + 4):
        r_row, g_row, b_row, alpha_row = [], [], [], []
        for j in range(height // 2 - 4, height // 2 + 4):
            if len(pixels[i, j]) == 3:
                r, g, b = pixels[i, j]
                alpha = None
            elif len(pixels[i, j]) == 4:
                r, g, b, alpha = pixels[i, j]
            r_row.append(r)
            g_row.append(g)
            b_row.append(b)
            alpha_row.append(alpha)
        r_arr.append(r_row)
        g_arr.append(g_row)
        b_arr.append(b_row)
        alpha_arr.append(alpha_row)
    r_arr_mul = mark_func(r_arr)
    g_arr_mul = mark_func(g_arr)
    b_arr_mul = mark_func(b_arr)
    def row_to_line(row):
        return "[" + ", ".join(str(round(x, 0)) for x in row) + "]"

    data_json_ready = {
        "r": [row_to_line(row) for row in r_arr_mul],
        "g": [row_to_line(row) for row in g_arr_mul],
        "b": [row_to_line(row) for row in b_arr_mul],
    }

    with open("mark_center_dct.json", "w", encoding="utf-8") as f:
        json.dump(data_json_ready, f, ensure_ascii=False, indent=4)  
    for matr in [r_arr_mul, g_arr_mul, b_arr_mul]:
        for coord in marker_check_coords:
            matr[coord[0]][coord[1]] = dct_marker
   
    r_arr = mark_func_reverse(r_arr_mul)
    g_arr = mark_func_reverse(g_arr_mul)
    b_arr = mark_func_reverse(b_arr_mul)
    for di, i in enumerate(range(width // 2 - 4, width // 2 + 4)):
        for dj, j in enumerate(range(height // 2 - 4, height // 2 + 4)):
            if len(pixels[i, j]) == 3:
                pixels[i, j] = (r_arr[di][dj], g_arr[di][dj], b_arr[di][dj])
            elif len(pixels[i, j]) == 4:
                pixels[i, j] = (r_arr[di][dj], g_arr[di][dj], b_arr[di][dj], alpha_arr[di][dj])

    return pixels
def spiral_block_coords(wb, hb, start_r, start_c):
    r, c = start_r, start_c
    total = wb * hb
    yielded = 0
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]  
    step_len = 1
    dir_idx = 0
    yielded += 1  
    while yielded <= total:
        for _ in range(2):
            dr, dc = dirs[dir_idx % 4]
            for _ in range(step_len):
                r += dr
                c += dc
                if 0 <= r < hb and 0 <= c < wb:
                    yield (r, c)
                    yielded += 1
                    if yielded > total:
                        return
            dir_idx += 1
        step_len += 1

def linear_block_coords(wb, hb):
    for r in range(wb):
        for c in range(hb):
            yield (r, c)
def inject(img, redDct, greenDct, blueDct, alphaMat, mes, k1, k2, func, funcReverse, mode: str = 'lsb'):
    pix = img.load()
    img_copy = img.copy()
    pix_copy = img_copy.load()
    black_block = np.zeros((8, 8)).astype(int)
    wc, hc = img.size
    block_size = 8
    wb = wc // block_size
    hb = hc // block_size
    start_x = wc // 2 - 4
    start_y = hc // 2 - 4
    start_block_x = start_x // block_size
    start_block_y = start_y // block_size

    red_blocks = []
    green_blocks = []
    blue_blocks = []
    c = 0
    skip = 0 
    if mode == 'lsb':
        coords = spiral_block_coords(wb, hb, start_block_y, start_block_x)
        pix = mark_center(wc, hc, pix, func, funcReverse)

    else:
        coords = linear_block_coords(wb, hb)
    for br, bc in coords:
        skip +=1
        if c >= len(mes):
            break
        if skip % 4 !=0:
            continue
        ic = bc * block_size
        jc = br * block_size
        mat1r, mat1g, mat1b, mat1a = [], [], [], []
        for i in range(ic, ic + block_size):
            mat2r, mat2g, mat2b, mat2a = [], [], [], []
            for j in range(jc, jc + block_size):
                if len(pix[i, j]) == 3:
                    c0, c1, c2 = pix[i, j]
                    alpha = None
                elif len(pix[i, j]) == 4:
                    c0, c1, c2, alpha = pix[i, j]
                mat2r.append(c0)
                mat2g.append(c1)
                mat2b.append(c2)
                mat2a.append(alpha)
            mat1r.append(mat2r)
            mat1g.append(mat2g)
            mat1b.append(mat2b)
            mat1a.append(mat2a)
        matr = np.array(mat1r, dtype=np.uint8)
        matg = np.array(mat1g, dtype=np.uint8)
        matb = np.array(mat1b, dtype=np.uint8)
        try:
            mata = np.array(mat1a, dtype=np.uint8)
        except:
            mata = np.array(mat1a)
        redHad = func(matr)
        greenHad = func(matg)
        blueHad = func(matb)
        alphaMat = mata 
        if mes[c] == '0':
            redInjected = injectMat0(redHad, k1, k2)
            greenInjected = injectMat0(greenHad, k1, k2)
            blueInjected = injectMat0(blueHad, k1, k2)
        if mes[c] == '1':
            redInjected = injectMat1(redHad, k1, k2)
            greenInjected = injectMat1(greenHad, k1, k2)
            blueInjected = injectMat1(blueHad, k1, k2)
        redReverse = funcReverse(redInjected)
        greenReverse = funcReverse(greenInjected)
        blueReverse = funcReverse(blueInjected)
        pix = replace_block(ic, jc, redReverse, greenReverse, blueReverse, alphaMat, pix)
        pix_copy = replace_block(ic, jc, black_block, black_block, black_block, alphaMat, pix_copy)
        red_blocks.append(redReverse)
        green_blocks.append(greenReverse)
        blue_blocks.append(blueReverse)
        c += 1
    print('save')
    img_copy.save('inject_map.bmp')

    save_matrices_pretty(red_blocks, "inject_matrices_r.json")
    save_matrices_pretty(green_blocks, "inject_matrices_g.json")
    save_matrices_pretty(blue_blocks, "inject_matrices_b.json")
    return img

def to_list2d(matrix):
    return [[int(x) for x in row] for row in matrix]

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

def find_reference_block_dct(width, height, pixels):
    best_score = float("inf")
    best_coords = (None, None)
    for i in range(height - 7):
        for j in range(width - 7):

            block_r = np.zeros((8, 8), dtype=float)
            block_g = np.zeros((8, 8), dtype=float)
            block_b = np.zeros((8, 8), dtype=float)

            for y in range(8):
                for x in range(8):
                    px = pixels[j + x, i + y]
                    if len(px) == 3:
                        r, g, b = px
                    else:
                        r, g, b, _ = px

                    block_r[y, x] = r
                    block_g[y, x] = g
                    block_b[y, x] = b

            dct_r = mark_func(block_r)
            dct_g = mark_func(block_g)
            dct_b = mark_func(block_b)


            score = 0.0

            for coord in marker_check_coords:
                cy, cx = coord
                score += abs(dct_r[cy][cx] - dct_marker)
                score += abs(dct_g[cy][cx] - dct_marker)
                score += abs(dct_b[cy][cx] - dct_marker)

            if score < best_score:
                best_score = score
                best_coords = (j, i)
    return best_coords

def extract(img, redDct, greenDct, blueDct, k1, k2, l, func, mode: str = 'lsb'):
    wc, hc = img.size
    block_size = 8
    wb = wc // block_size
    hb = hc // block_size
    start_x = wc // 2 - 4
    start_y = hc // 2 - 4
    start_block_x = start_x // block_size
    start_block_y = start_y // block_size
    mes = ""
    red_mes = ''
    green_mes = ''
    blue_mes = ''
    c = 0
    skip = 0
    red_blocks = []
    green_blocks = []
    blue_blocks = []
    ref_x, ref_y = find_reference_block_dct(wc, hc, pix)
    print('find_ref_block', ref_x, ref_y)
    if ref_x == None or ref_y == None:
        print('Не удалось найти блок отсчёта, используется значение по-умолчанию')
        ref_x = start_x
        ref_y = start_y

    if mode == 'lsb':
        coords = spiral_block_coords(wb, hb, ref_y // block_size, ref_x // block_size)
    else:
        coords = linear_block_coords(wb, hb)
    for br, bc in coords:
        skip +=1
        if skip % 4 !=0:
            continue
        if c >= l:
            break
        ic = bc * block_size
        jc = br * block_size

        mat1r, mat1g, mat1b = [], [], []
        for i in range(ic, ic + block_size):
            mat2r, mat2g, mat2b = [], [], []
            for j in range(jc, jc + block_size):
                if len(pix[i, j]) == 3:
                    c0, c1, c2 = pix[i, j]
                elif len(pix[i, j]) == 4:
                    c0, c1, c2, alpha = pix[i, j]
                mat2r.append(c0)
                mat2g.append(c1)
                mat2b.append(c2)
            mat1r.append(mat2r)
            mat1g.append(mat2g)
            mat1b.append(mat2b)
        matr = np.array(mat1r, dtype=np.uint8)
        matg = np.array(mat1g, dtype=np.uint8)
        matb = np.array(mat1b, dtype=np.uint8)
        redHad = func(matr)
        greenHad = func(matg)
        blueHad = func(matb)
        c_bit = 0
        
        if abs(redHad[k1[0]][k1[1]]) > abs(redHad[k2[0]][k2[1]]): 
            red_mes += '1'
            c_bit+=1
        else:
            red_mes += '0'
        if abs(greenHad[k1[0]][k1[1]]) > abs(greenHad[k2[0]][k2[1]]): 
            green_mes += '1'
            c_bit += 1
        else:
            green_mes += '0'
        if abs(blueHad[k1[0]][k1[1]]) > abs(blueHad[k2[0]][k2[1]]): 
            blue_mes += '1'
            c_bit += 1
        else:
            blue_mes += '0'
        if c_bit>=2:
            mes += '1'
        else:
            mes += '0'
        c+=1
        red_blocks.append(matr)
        green_blocks.append(matg)
        blue_blocks.append(matb)
    return mes


def readImage(img, func):
    wc, hc = img.size
    global pix
    pix = img.load()
    redDct, greenDct, blueDct, alphaMat = convertPixToDCT(pix, wc, hc, func)
    return redDct, greenDct, blueDct, alphaMat
def convertPixToDCT(pix, wc, hc, func):
    redDct = []
    greenDct = []
    blueDct = []
    alphaMat = []
    i, j, ic, jc = 0, 0, 0, 0
    while ic + 8 <= wc:
        while jc + 8 <= hc:
            mat1r = []
            mat1g = []
            mat1b = []
            mat1a = []
            for i in range(ic, ic + 8):
                mat2r = []
                mat2g = []
                mat2b = []
                mat2a = []
                for j in range(jc, jc + 8):
                    if len(pix[i, j]) == 3:
                        c0, c1, c2 = pix[i, j] 
                        alpha = None 
                    elif len(pix[i, j]) == 4:
                        c0, c1, c2, alpha = pix[i, j]
                    mat2r.append(c0)
                    mat2g.append(c1)
                    mat2b.append(c2)
                    mat2a.append(alpha)
                mat1r.append(mat2r)
                mat1g.append(mat2g)
                mat1b.append(mat2b)
                mat1a.append(mat2a)
            matr = np.array(mat1r)
            matg = np.array(mat1g)
            matb = np.array(mat1b)
            mata = np.array(mat1a)
            redDct.append(func(matr))
            greenDct.append(func(matg))
            blueDct.append(func(matb))
            alphaMat.append(mata)
            jc += 8
        jc = 0
        ic += 8
    return redDct, greenDct, blueDct, alphaMat
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

def save_matrices_pretty(matrices, filename):
    def to_list2d(matrix):
        return [[float(x) for x in row] for row in matrix]

    with open(filename, "w", encoding="utf-8") as f:
        f.write("[\n")
        for i, matrix in enumerate(matrices):
            mat = to_list2d(matrix)
            f.write("  [\n")
            for j, row in enumerate(mat):
                f.write("    " + str(row))
                if j < len(mat) - 1:
                    f.write(",")
                f.write("\n")
            f.write("  ]")
            if i < len(matrices) - 1:
                f.write(",")
            f.write("\n")
        f.write("]\n")