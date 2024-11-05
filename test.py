import numpy as np
from scipy.linalg import hadamard
import koch
import numpy as np
import time
from scipy.fftpack import dct
import math
# from progress.bar import IncrementalBar
# Создаем матрицу Адамара
def dct_matrix(N):
    C = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == 0:
                C[i, j] = np.sqrt(1 / N)
            else:
                C[i, j] = np.sqrt(2 / N) * np.cos((np.pi * (2 * j + 1) * i) / (2 * N))
    return C
size = 8
hadamard_matrix = hadamard(size)/np.sqrt(size)
dct =  dct_matrix(size)
def dctMul(mat):
    mat1 = np.matmul(np.transpose(dct), mat)
    return np.matmul(mat1, dct)
def dctMulReverse(mat):
    mat1 = np.matmul(dct, mat)
    return (np.matmul(mat1, np.transpose(dct))).astype(int)
def hadMul(mat):
    mat1 = np.matmul(np.transpose(hadamard_matrix), mat)
    return (np.matmul(mat1, hadamard_matrix))
def hadMulReverse(mat):
    mat1 = np.matmul(hadamard_matrix, mat)
    return (np.matmul(mat1, np.transpose(hadamard_matrix))).astype(int)



matrix = np.random.randint(0, 256, size=(size, size))
# for row in formatted_matrix:
#     print(row)

# start = time.time()
# hadMulReverse(hadMul(matrix))
# end = time.time()
# print(end-start)

difS = []
difL = []
dif = []
m = np.random.randint(0, 256, size=(size, size))
print(m)
print(dctMulReverse(dctMul(m))==m)
# print(hadMulReverse(hadMul(m))==m)
print(hadMulReverse(hadMul(m))==m)
for _ in range(10):
    matrix = []
    for _ in range(32400):
        matrix.append(np.random.randint(0, 256, size=(size, size)))
    start = time.time()
    for m in matrix:
        dctMulReverse(dctMul(m))
    end = time.time()
    dif2 = end - start
    # print(start)
    # print(end)
    # print(f"{dif1:.4}")
    start2 = time.time()
    for m in matrix:
        hadMulReverse(hadMul(m))
    end2 = time.time()
    dif1 = end2 - start2
    # print(start)
    # print(end)
    # print(f"{dif2:.4}")
    if dif2 >= dif1:
        #когда дкт дольше
        difS.append(dif2-dif1)
    else:
        # когда адамра дольше
        difL.append(dif2-dif1)
    dif.append(dif2-dif1)
print(sum(dif)/len(dif))
print(len(difS))
print(len(difL))

# print(matrix-hadMulReverse(hadMul(matrix)))
# print(matrix-dctMulReverse(dctMul(matrix)))
# print(str.replace("0", "").replace("1", "") == "")
# for i in range(5):
#     # Создаем матрицу 8x8 с элементами от 0 до 255
#     matrix = np.random.randint(0, 256, size=(size, size))
#     # Умножаем матрицы
#     result = np.matmul(matrix, hadamard_matrix)
#     print("Исходная матрица:")
#     print(matrix)
#     print("После умножения:")
#     print(result)
