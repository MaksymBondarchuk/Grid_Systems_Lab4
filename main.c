#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "linalg.h"

/* Ім'я вхідного файлу */
const char *input_file_MA = "MA.txt";
const char *input_file_b = "b.txt";
const char *output_file_x = "x.txt";

/* Тег повідомленя, що містить стовпець матриці */
const int COLUMN_TAG = 0x1;

void print_matrix(struct my_matrix *matrix) {
    for (int i = 0; i < matrix->rows; ++i) {
        for (int j = 0; j < matrix->cols; ++j) {
            printf("%4.2f ", matrix->data[i * matrix->rows + j]);
        }
        printf("\n");
    }
}

void print_vector(struct my_vector *vector) {
    for (int i = 0; i < vector->size; ++i) {
        printf("%4.2f ", vector->data[i]);
    }
    printf("\n");
}

/* Основна функція (програма обчислення визначника) */
int main(int argc, char *argv[]) {
    /* Ініціалізація MPI */
    MPI_Init(&argc, &argv);

    /* Отримання загальної кількості задач та рангу поточної задачі */
    int tasks_number, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &tasks_number);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Зчитування даних в задачі 0 */
    struct my_matrix *MA;
    struct my_vector *b;
    int N;
    if (rank == 0) {
        MA = read_matrix(input_file_MA);
        b = read_vector(input_file_b);

        if (MA->rows != MA->cols) {
            fatal_error("Matrix is not square!", 4);
        }
        if (MA->rows != b->size) {
            fatal_error("Dimensions of matrix and vector don’t match!", 5);
        }
        N = MA->rows;
    }

    /* Розсилка всім задачам розмірності матриць та векторів */
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Обчислення кількості стовпців, які будуть зберігатися в кожній задачі та
     * виділення пам'яті для їх зберігання */
    int rows_count_for_one_task = N / tasks_number;
    struct my_matrix *MAh = matrix_alloc(N, rows_count_for_one_task, .0);

    /* Створення та реєстрація типу даних для стовпця елементів матриці */
    MPI_Datatype matrix_columns;
    MPI_Type_vector(N * rows_count_for_one_task, 1, tasks_number, MPI_DOUBLE, &matrix_columns);
    MPI_Type_commit(&matrix_columns);

    /* Створення та реєстрація типу даних для структури вектора */
    MPI_Datatype vector_struct;
    MPI_Aint extent;
    MPI_Type_extent(MPI_INT, &extent);            // визначення розміру в байтах
    MPI_Aint offsets[] = {0, extent};
    int lengths[] = {1, N + 1};
    MPI_Datatype oldtypes[] = {MPI_INT, MPI_DOUBLE};
    MPI_Type_struct(2, lengths, offsets, oldtypes, &vector_struct);
    MPI_Type_commit(&vector_struct);

    // Calculate L and U matrices
    struct my_matrix *ML;
    if (rank == 0) {
        ML = matrix_alloc(N, N, 0.0);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                ML->data[i * N + j] = MA->data[i * N + j];
            }
        }
        printf("Generated ML matrix\n");

        // Formulas from manual (3.2)
        for (int k = 0; k < N - 1; k++) {
            // ML ik     i = 1..k-1                  row 1
            for (int i = 0; i < k; ++i) {
                ML->data[i * N + k] = 0;
            }

            // ML kk                                 row 2
            ML->data[k * N + k] = 1;

            // ML ik     i = k+1..N                  row3
            for (int i = k + 1; i < N; ++i) {
                if (MA->data[k * N + k] == 0) {
                    ML->data[i * N + k] = 0;
                } else {
                    ML->data[i * N + k] = MA->data[i * N + k] / MA->data[k * N + k];
                }
//                printf("k=%d, i=%d, 1...=%f\n", k, i, MA->data[i * N + k]);
//                printf("k=%d, i=%d, 2...=%f\n", k, i, MA->data[k * N + k]);
            }

            // MA ij     i = k+1..N, j = 1..k        row 5
            for (int i = k; i < N; ++i) {
                for (int j = 0; j < k; ++j) {
                    MA->data[i * N + j] = 0;
                }
            }

            // MA ij     i = k+1..N, j = k+1..N      row 6
            for (int i = k + 1; i < N; ++i) {
                for (int j = k; j < N; ++j) {
                    MA->data[i * N + j] -= ML->data[i * N + j] * MA->data[k * N + j];
                }
            }

            printf("\nk=%d\n", k);
            printf("ML\n");
            print_matrix(ML);

            printf("\nMA\n");
            print_matrix(MA);
        }

        // Finish ML calculation
        for (int i = 0; i < N - 1; ++i) {
            ML->data[i * N + N - 1] = 0;
        }
        ML->data[(N - 1) * N + N - 1] = 1;

        printf("ML\n");
        print_matrix(ML);

        printf("\nMA\n");
        print_matrix(MA);

        struct my_vector *y = vector_alloc(N, 0.0);
        for (int i = 0; i < N; ++i) {
            double sum = 0;
            for (int j = 0; j < i; ++j) {
                sum += y->data[j] * ML->data[i * N + j];
            }
            y->data[i] = b->data[i] - sum;
        }
        struct my_vector *x = vector_alloc(N, 0.0);
        for (int i = N - 1; 0 <= i; i--) {
            double sum = 0;
            for (int j = i + 1; j < N; ++j) {
                sum += x->data[j] * MA->data[i * N + j];
            }
            x->data[i] = (y->data[i] - sum) / (MA->data[i * N + i]);
        }
        printf("\ny\n");
        print_vector(y);
        printf("\nx\n");
        print_vector(x);
    }

    /* Розсилка стовпців матриці з задачі 0 в інші задачі */
    if (rank == 0) {
        for (int i = 1; i < tasks_number; i++) {
            MPI_Send(&(MA->data[i]), 1, matrix_columns, i, COLUMN_TAG, MPI_COMM_WORLD);
        }
        /* Копіювання елементів стовпців даної задачі */
        for (int i = 0; i < rows_count_for_one_task; i++) {
            int col_index = i * tasks_number;
            for (int j = 0; j < N; j++) {
                MAh->data[j * rows_count_for_one_task + i] = MA->data[j * N + col_index];
            }
        }
        free(MA);
    } else {
        MPI_Recv(MAh->data, N * rows_count_for_one_task, MPI_DOUBLE, 0, COLUMN_TAG, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }

    /* Поточне значення вектору l_i */
    struct my_vector *current_l = vector_alloc(N, .0);
    /* Частина стовпців матриці L */
    struct my_matrix *MLh = matrix_alloc(N, rows_count_for_one_task, .0);

    /* Основний цикл ітерації (кроки) */
    for (int step = 0; step < N - 1; step++) {
        /* Вибір задачі, що містить стовпець з ведучім елементом та обчислення
         * поточних значень вектору l_i */
        if (step % tasks_number == rank) {
            int col_index = (step - (step % tasks_number)) / tasks_number;
            MLh->data[step * rows_count_for_one_task + col_index] = 1.;
            for (int i = step + 1; i < N; i++) {
                MLh->data[i * rows_count_for_one_task + col_index] =
                        MAh->data[i * rows_count_for_one_task + col_index] /
                        MAh->data[step * rows_count_for_one_task + col_index];
            }
            for (int i = 0; i < N; i++) {
                current_l->data[i] = MLh->data[i * rows_count_for_one_task + col_index];
            }
        }
        /* Розсилка поточних значень l_i */
        MPI_Bcast(current_l, 1, vector_struct, step % tasks_number, MPI_COMM_WORLD);

        /* Модифікація стовпців матриці МА відповідно до поточного l_i */
        for (int i = step + 1; i < N; i++) {
            for (int j = 0; j < rows_count_for_one_task; j++) {
                MAh->data[i * rows_count_for_one_task + j] -=
                        MAh->data[step * rows_count_for_one_task + j] * current_l->data[i];
            }
        }
    }

    /* Обислення добутку елементів, які знаходяться на головній діагоналі
     * основної матриці (з урахуванням номеру стовпця в задачі) */
    double prod = 1.;
    for (int i = 0; i < rows_count_for_one_task; i++) {
        int row_index = i * tasks_number + rank;
        prod *= MAh->data[row_index * rows_count_for_one_task + i];
        printf("prod=%f\n", prod);
    }

    /* Згортка добутків елементів головної діагоналі та вивід результату в задачі 0 */
    if (rank == 0) {
        MPI_Reduce(MPI_IN_PLACE, &prod, 1, MPI_DOUBLE, MPI_PROD, 0, MPI_COMM_WORLD);
        printf("prod=%f\n", prod);
        printf("%lf", prod);
    } else {
        MPI_Reduce(&prod, NULL, 1, MPI_DOUBLE, MPI_PROD, 0, MPI_COMM_WORLD);
    }

    /* Повернення виділених ресурсів */
    MPI_Type_free(&matrix_columns);
    MPI_Type_free(&vector_struct);
    return MPI_Finalize();
}

