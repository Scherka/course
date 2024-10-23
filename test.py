import numpy as np
from scipy.linalg import hadamard
import koch
import numpy as np
import time
# from progress.bar import IncrementalBar
# Создаем матрицу Адамара
size = 8
hadamard_matrix = hadamard(size)
dct = koch.dctCreate()
def dctMul(mat):
    mat1 = np.matmul(np.transpose(dct), mat)
    return np.matmul(mat1, dct)
def dctMulReverse(mat):
    mat1 = np.matmul(dct, mat)
    return np.matmul(mat1, np.transpose(dct)).astype(int)
def hadMul(mat):
    mat1 = np.matmul(np.transpose(hadamard_matrix), mat)
    return (np.matmul(mat1, hadamard_matrix))
def hadMulReverse(mat):
    mat1 = np.matmul(hadamard_matrix, mat)
    return (np.matmul(mat1, np.transpose(hadamard_matrix))).astype(int)


print(f'{31:05b}')
# matrix = np.random.randint(0, 256, size=(size, size))
# dct = dctMul(matrix)
# had = hadMul(dct)
# formatted_matrix = [[f"{num:.2f}" for num in row] for row in had]
# for row in formatted_matrix:
#     print(row)

# start = time.time()
# hadMulReverse(hadMul(matrix))
# end = time.time()
# print(end-start)

# difS = []
# difL = []
# dif = []
# for _ in range(10):
#     matrix = np.random.randint(0, 256, size=(size, size))
#     start = time.time()
#     for _ in range(32400):
#         dctMulReverse(dctMul(matrix))
#     end = time.time()
#     dif2 = end - start
#     # print(start)
#     # print(end)
#     # print(f"{dif1:.4}")
#     start = time.time()
#     for _ in range(32400):
#         hadMulReverse(hadMul(matrix))
#     # end = time.time()
#     dif1 = end - start
#     # print(start)
#     # print(end)
#     # print(f"{dif2:.4}")
#     if dif2 >= dif1:
#         difS.append(dif2-dif1)
#     else:
#         difL.append(dif2-dif1)
#     dif.append(dif2-dif1)
# print(sum(dif)/len(dif))
# print(len(difS))
# print(len(difL))

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
