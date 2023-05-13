import random
import sys
import time


def get_matrix_lenght():
    arg1 = sys.argv[1]
    return int(arg1)


def generate_matrix(lenght):
    m_a = [[random.random() for _ in range(lenght)] for _ in range(lenght)]
    m_b = [[random.random() for _ in range(lenght)] for _ in range(lenght)]

    return m_a, m_b


def multiply_matrix(matrix_a, matrix_b, lenght):
    matrix_c = [[0 for _ in range(lenght)] for _ in range(lenght)]

    for i in range(lenght):
        for j in range(lenght):
            for k in range(lenght):
                matrix_c[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return matrix_c


def main():
    lenght = get_matrix_lenght()
    matrix_a, matrix_b = generate_matrix(lenght)
    init = time.perf_counter()
    _ = multiply_matrix(matrix_a, matrix_b, lenght)
    final = time.perf_counter()
    print(f"Time to calculate matrix {lenght}x{lenght}: {final - init:0.4f}s")


if __name__ == "__main__":
    main()
