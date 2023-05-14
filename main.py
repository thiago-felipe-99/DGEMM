import random
import sys
import time


def get_matrix_length():
    arg1 = sys.argv[1]
    return int(arg1)


def generate_matrix(length):
    m_a = [[random.random() for _ in range(length)] for _ in range(length)]
    m_b = [[random.random() for _ in range(length)] for _ in range(length)]

    return m_a, m_b


def multiply_matrix(matrix_a, matrix_b, length):
    matrix_c = [[0 for _ in range(length)] for _ in range(length)]

    for i in range(length):
        for j in range(length):
            for k in range(length):
                matrix_c[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return matrix_c


def main():
    length = get_matrix_length()
    matrix_a, matrix_b = generate_matrix(length)
    init = time.perf_counter()
    _ = multiply_matrix(matrix_a, matrix_b, length)
    final = time.perf_counter()
    print(f"Time to calculate matrix {length}x{length}: {final - init:0.4f}s")


if __name__ == "__main__":
    main()
