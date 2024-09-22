import numpy as np
from scipy.linalg import hadamard


# Создаем матрицу Адамара
size = 8
hadamard_matrix = hadamard(size)


for i in range(5):
    # Создаем матрицу 8x8 с элементами от 0 до 255
    matrix = np.random.randint(0, 256, size=(size, size))
    # Умножаем матрицы
    result = np.matmul(matrix, hadamard_matrix)
    print("Исходная матрица:")
    print(matrix)
    print("После умножения:")
    print(result)
