import numpy as np
import time
from numba import jit
import math
import random

# Чистый Python
def softmax_python(matrix):
    if not matrix:
        return []

    result = []
    for row in matrix:
        exp_values = [math.exp(x) for x in row]
        sum_exp = sum(exp_values)
        result.append([x / sum_exp for x in exp_values])
    return result

# Numba версия
@jit(nopython=True)
def softmax_numba(matrix):
    n_rows = matrix.shape[0]
    n_cols = matrix.shape[1]
    result = np.zeros((n_rows, n_cols))
    
    for i in range(n_rows):
        sum_exp = 0.0
        for j in range(n_cols):
            result[i, j] = math.exp(matrix[i, j])
            sum_exp += result[i, j]
        
        for j in range(n_cols):
            result[i, j] /= sum_exp
    
    return result

# NumPy версия
def softmax_numpy(matrix):

    exp_matrix = np.exp(matrix)
    return exp_matrix / np.sum(exp_matrix, axis=-1, keepdims=True)

# Генерация матриц
def generate_test_matrix(size):
    np.random.seed(42)
    random.seed(42)
    matrix_np = np.random.uniform(-10.0, 10.0, (size, size))
    matrix_python = matrix_np.tolist()
    return matrix_python, matrix_np

# Тестирование
def test_softmax_python(matrix):
    start_time = time.time()
    result = softmax_python(matrix)
    end_time = time.time()
    row_sums = [sum(row) for row in result]
    avg_error = sum(abs(1.0 - s) for s in row_sums) / len(row_sums)
    return end_time - start_time, avg_error

def test_softmax_numba(matrix):
    small_matrix = np.random.uniform(-10.0, 10.0, (10, 10))
    softmax_numba(small_matrix)
    
    start_time = time.time()
    result = softmax_numba(matrix)
    end_time = time.time()
    row_sums = np.sum(result, axis=1)
    avg_error = np.mean(np.abs(1.0 - row_sums))
    return end_time - start_time, avg_error

def test_softmax_numpy(matrix):
    start_time = time.time()
    result = softmax_numpy(matrix)
    end_time = time.time()
    row_sums = np.sum(result, axis=1)
    avg_error = np.mean(np.abs(1.0 - row_sums))
    return end_time - start_time, avg_error

def main():
    size = int(input("Введите размер квадратной матрицы: "))
    
    matrix_python, matrix_np = generate_test_matrix(size)
    
    print("РЕЗУЛЬТАТЫ ПРОИЗВОДИТЕЛЬНОСТИ:")
    time_python, error_python = test_softmax_python(matrix_python)
    print(f"Чистый Python:")
    print(f"  Время: {time_python:.6f} сек")
    print(f"  Средняя ошибка суммы: {error_python:.10f}")
    print("-" * 50)
    
    time_numba, error_numba = test_softmax_numba(matrix_np)
    print(f"Numba:")
    print(f"  Время: {time_numba:.6f} сек")
    print(f"  Средняя ошибка суммы: {error_numba:.10f}")
    print("-" * 50)
    
    time_numpy, error_numpy = test_softmax_numpy(matrix_np)
    print(f"NumPy:")
    print(f"  Время: {time_numpy:.6f} сек")
    print(f"  Средняя ошибка суммы: {error_numpy:.10f}")
    print("-" * 50)

if __name__ == "__main__":
    main()
