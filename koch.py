from PIL import Image, ImageDraw
import numpy as np
from scipy.linalg import hadamard
eps = 50
coefs = [[0, 6], [0, 7], [1, 6], [1, 7], [2, 4], [2, 5], [3, 4], [3, 5], [4, 2], [4, 3], [5, 2], [5, 3], [6, 0], [6, 1],
         [7, 0], [7, 1]]
ls_bit = 5 # наименее значимый бит для маркировки центра
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
# замена блока пикселей изображения
def replace_block(i_start, j_start, redMatrix, greenMatrix, blueMatrix, pixels):
    block_size = 8
    for i in range(block_size):
        for j in range(block_size):
            pixels[i_start + i, j_start + j] = (redMatrix[i, j], greenMatrix[i, j], blueMatrix[i, j])
    return pixels

def mark_center(width, height, pixels): 
    block_size = 8 
    val =2 ** ls_bit
    print('mark_center',width //2 - 4, height //2 - 4 ) # блок отсчёта 
    for i in range(width //2 - 4, width //2 + 4): 
        for j in range(height //2 - 4, height //2 + 4): 
            r, g, b = pix[i, j] 
            r = r // val * val + val - 1
            g = g // val * val + val - 1
            b = b // val * val + val - 1
            # r = r // val * val 
            # g = g // val * val 
            # b = b // val * val
            pixels[i, j] = (r, g, b) 
            # pixels[i, j] = (0, 0, 0) 
    return pixels
def mark_center1(width, height, pixels, func, funcReverse):
    block_size = 8
    print('mark_center', width // 2 - 4, height // 2 - 4)

    r_arr, g_arr, b_arr = [], [], []

    # collect modified values
    for i in range(width // 2 - 4, width // 2 + 4):
        r_row, g_row, b_row = [], [], []
        for j in range(height // 2 - 4, height // 2 + 4):
            r, g, b = pixels[i, j]
            r_row.append(r)
            g_row.append(g)
            b_row.append(b)
        r_arr.append(r_row)
        g_arr.append(g_row)
        b_arr.append(b_row)
    r_arr_mul = func(r_arr)
    g_arr_mul = func(g_arr)
    b_arr_mul = func(b_arr)
    for i in range(block_size):
        for j in range(block_size):
            r_arr_mul[i][j] = r_arr_mul[i][j] // 4 * 4 + 3
            g_arr_mul[i][j] = g_arr_mul[i][j] // 4 * 4 + 3
            b_arr_mul[i][j] = b_arr_mul[i][j] // 4 * 4 + 3
    r_arr = funcReverse(r_arr_mul)
    g_arr = funcReverse(g_arr_mul)
    b_arr = funcReverse(b_arr_mul)
    # reapply modified pixels
    for di, i in enumerate(range(width // 2 - 4, width // 2 + 4)):
        for dj, j in enumerate(range(height // 2 - 4, height // 2 + 4)):
            pixels[i, j] = (r_arr[di][dj], g_arr[di][dj], b_arr[di][dj])

    return pixels
def spiral_block_coords(wb, hb, start_r, start_c):
    """Генерирует координаты блоков по спирали от точки (start_r, start_c)."""
    r, c = start_r, start_c
    total = wb * hb
    yielded = 0
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # вправо, вниз, влево, вверх
    step_len = 1
    dir_idx = 0

    # первый блок (центр) не используем, но стартуем от него
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


def inject(img, redDct, greenDct, blueDct, mes, k1, k2, func, funcReverse):
    pix = img.load()
    wc, hc = img.size
    block_size = 8
    wb = wc // block_size
    hb = hc // block_size

    # центр как точка отсчёта
    start_x = wc // 2 - 4
    start_y = hc // 2 - 4
    start_block_x = start_x // block_size
    start_block_y = start_y // block_size

    red_blocks = []
    green_blocks = []
    blue_blocks = []
    # маркировка центра с помошью lsb
    pix = mark_center(wc, hc, pix)
    # pix = mark_center(wc, hc, pix, func, funcReverse)


    c = 0
    skip = 0 
    # # заменяем исходный блок на ч/б# спиральный обход блоков (центр — точка отсчёта, но не используется)
    for br, bc in spiral_block_coords(wb, hb, start_block_y, start_block_x):
        skip +=1
        if c >= len(mes):
            break
        if skip % 4 !=0:
            continue
        # print(br, bc)
        ic = bc * block_size
        jc = br * block_size
        mat1r, mat1g, mat1b = [], [], []
        for i in range(ic, ic + block_size):
            mat2r, mat2g, mat2b = [], [], []
            for j in range(jc, jc + block_size):
                c0, c1, c2 = pix[i, j]
                mat2r.append(c0)
                mat2g.append(c1)
                mat2b.append(c2)
            mat1r.append(mat2r)
            mat1g.append(mat2g)
            mat1b.append(mat2b)
        # создаём numpy-массивы каналов
        matr = np.array(mat1r, dtype=np.uint8)
        matg = np.array(mat1g, dtype=np.uint8)
        matb = np.array(mat1b, dtype=np.uint8)
        redHad = func(matr)
        greenHad = func(matg)
        blueHad = func(matb)
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
        pix = replace_block(ic, jc, redReverse, greenReverse, blueReverse, pix)

        c += 1
    print('save')
    # img.save('gray.bmp')
    return img

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
def check_center(width, height, pixels):
    block_size = 8
    val = 2 ** ls_bit
    # блок отсчёта
    print('check_center',width, height)
    red_block = []
    green_block = []
    blue_block = []
    for i in range(width, width + 8):
        red_row = []
        green_row = []
        blue_row = []
        for j in range(height, height + 8):
            r, g, b = pixels[i, j]
            red_row.append(bin(r))
            green_row.append(bin(g))
            blue_row.append(bin(b))
        red_block.append(red_row)
        green_block.append(green_row)
        blue_block.append(blue_row)
    print('check center')
    print(red_block)
    print(green_block)
    print(blue_block)
def find_reference_block(width, height, pixels):
    val = 2 ** ls_bit
    t_hold = 2 ** (ls_bit-1)
    for i in range(height - 7):
        for j in range(width - 7):
            all_end_with_1 = True
            redm = []
            greenm =[]
            bluem = []
            for y in range(8):
                redr = []
                greenr =[]
                bluer = []
                for x in range(8):
                    r, g, b = pixels[j + x, i + y]
                    redr.append(r), greenr.append(g), bluer.append(b)
                    if (r % val < t_hold) or (g % val < t_hold) or (b % val < t_hold):
                    # if (r % val > 0) or (g % val > 0) or (b % val > 0):
                        all_end_with_1 = False
                        break
                redm.append(redr), greenm.append(greenr), bluem.append(bluer)
                if not all_end_with_1:
                    break
            if all_end_with_1:
                print(redm)
                print(greenm)
                print(bluem)
                return (j, i)  # top-left corner of found 8×8 block
    return None, None  # not found
def find_reference_block1(width, height, pixels, func):
    block_size = 8
    for i in range(0, height - block_size + 1):
        for j in range(0, width - block_size + 1):
            # extract block
            r_arr, g_arr, b_arr = [], [], []
            for y in range(block_size):
                r_row, g_row, b_row = [], [], []
                for x in range(block_size):
                    r, g, b = pixels[j + x, i + y]
                    r_row.append(r)
                    g_row.append(g)
                    b_row.append(b)
                r_arr.append(r_row)
                g_arr.append(g_row)
                b_arr.append(b_row)

            # apply transform
            r_t = func(r_arr)
            g_t = func(g_arr)
            b_t = func(b_arr)

            # check if this block has your marking pattern
            marked = True
            for y in range(block_size):
                for x in range(block_size):
                    if (r_t[y][x] % 4 != 3) or (g_t[y][x] % 4 != 3) or (b_t[y][x] % 4 != 3):
                        marked = False
                        break
                if not marked:
                    break

            if marked:
                return (j, i)

    return None, None

def extract(img, redDct, greenDct, blueDct, k1, k2, l, func):
    wc, hc = img.size
    block_size = 8
    wb = wc // block_size
    hb = hc // block_size
    start_x = wc // 2 - 4
    start_y = hc // 2 - 4
    start_block_x = start_x // block_size
    start_block_y = start_y // block_size
    mes = ""
    c = 0
    skip = 0
    check_center(start_x, start_y, pix)
    ref_x, ref_y = find_reference_block(wc, hc, pix)
    print('find_ref_block', ref_x, ref_y)
    if ref_x == None or ref_y == None:
        print('Не удалось найти блок отсчёта')
        return None
    # check_center(center_x, center_y, pix)
    # print(find_block_endswith1(wc, hc, pix), start_x, start_y)

    # # заменяем исходный блок на ч/б# спиральный обход блоков (центр — точка отсчёта, но не используется)
    for br, bc in spiral_block_coords(wb, hb, ref_y // block_size, ref_x // block_size):
        skip +=1
        if skip % 4 !=0:
            continue
        if c >= l:
            break
        # print(br, bc)
        ic = bc * block_size
        jc = br * block_size

        mat1r, mat1g, mat1b = [], [], []
        for i in range(ic, ic + block_size):
            mat2r, mat2g, mat2b = [], [], []
            for j in range(jc, jc + block_size):
                c0, c1, c2 = pix[i, j]
                mat2r.append(c0)
                mat2g.append(c1)
                mat2b.append(c2)
            mat1r.append(mat2r)
            mat1g.append(mat2g)
            mat1b.append(mat2b)
        # создаём numpy-массивы каналов
        matr = np.array(mat1r, dtype=np.uint8)
        matg = np.array(mat1g, dtype=np.uint8)
        matb = np.array(mat1b, dtype=np.uint8)
        redHad = func(matr)
        greenHad = func(matg)
        blueHad = func(matb)
        c_bit = 0
        if abs(redHad[k1[0]][k1[1]]) > abs(redHad[k2[0]][k2[1]]): c_bit+=1
        if abs(greenHad[k1[0]][k1[1]]) > abs(greenHad[k2[0]][k2[1]]): c_bit += 1
        if abs(blueHad[k1[0]][k1[1]]) > abs(blueHad[k2[0]][k2[1]]): c_bit += 1
        if c_bit>=2:
            mes += '1'
        else:
            mes += '0'
        c+=1
    # for k in range(l):
    #     if k%4==0:
    #         c_bit = 0
    #         if abs(redDct[k][k1[0]][k1[1]]) > abs(redDct[k][k2[0]][k2[1]]): c_bit+=1
    #         if abs(greenDct[k][k1[0]][k1[1]]) > abs(greenDct[k][k2[0]][k2[1]]): c_bit += 1
    #         if abs(blueDct[k][k1[0]][k1[1]]) > abs(blueDct[k][k2[0]][k2[1]]): c_bit += 1
    #         if c_bit>=2:
    #             mes += '1'
    #         else:
    #             mes += '0'
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
    wc, hc = img.size
    global pix
    pix = img.load()
    redDct, greenDct, blueDct = convertPixToDCT(pix, wc, hc, func)
    print('dct', len(redDct), len(redDct[0]))
    return redDct, greenDct, blueDct
def convertPixToDCT(pix, wc, hc, func):
    redDct = []
    greenDct = []
    blueDct = []
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