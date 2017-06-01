#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "linalg.h"

/* Input file name */
const char *input_file_MA = "MA.txt";

/* Tag of the message containing matrix column */
const int COLUMN_TAG = 0x1;

/* Main function (calculating the determinant problem) */
int main(int argc, char *argv[])
{
  /* Initialize MPI */
  MPI_Init(&argc, &argv);

  /* Get the total number of tasks and the current task rank */
  int np, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  /* Input data in task 0 */
  struct my_matrix *MA;
  int N;
  if(rank == 0)
  {
    MA = read_matrix(input_file_MA);

    if(MA->rows != MA->cols) {
      fatal_error("Matrix is not square!", 4);
    }
    N = MA->rows;
  }

  /* Send matrix and vector dimensions to all tasks */
  MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

  /* Calculate the number of columns to be stored in every task
   * and allocate memory for them */
  int part = N / np;
  struct my_matrix *MAh = matrix_alloc(N, part, .0);

  /* Create and register matrix column data type */
  MPI_Datatype matrix_columns;
  MPI_Type_vector(N*part, 1, np, MPI_DOUBLE, &matrix_columns);
  MPI_Type_commit(&matrix_columns);

  /* Create and register vector data type */
  MPI_Datatype vector_struct;
  MPI_Aint extent;
  MPI_Type_extent(MPI_INT, &extent);            // get size in bytes
  MPI_Aint offsets[] = {0, extent};
  int lengths[] = {1, N+1};
  MPI_Datatype oldtypes[] = {MPI_INT, MPI_DOUBLE};
  MPI_Type_struct(2, lengths, offsets, oldtypes, &vector_struct);
  MPI_Type_commit(&vector_struct);

  /* Send columns of the matrix from task 0 to other tasks */
  if(rank == 0)
  {
    for(int i = 1; i < np; i++)
    {
      MPI_Send(&(MA->data[i]), 1, matrix_columns, i, COLUMN_TAG, MPI_COMM_WORLD);
    }
    /* Copy elements of columns in the current task */
    for(int i = 0; i < part; i++)
    {
      int col_index = i*np;
      for(int j = 0; j < N; j++)
      {
        MAh->data[j*part + i] = MA->data[j*N + col_index];
      }
    }
    free(MA);
  }
  else
  {
    MPI_Recv(MAh->data, N*part, MPI_DOUBLE, 0, COLUMN_TAG, MPI_COMM_WORLD,
        MPI_STATUS_IGNORE);
  }

  /* Current value of the vector l_i */
  struct my_vector *current_l = vector_alloc(N, .0);
  /* Part of columns of the matrix L */
  struct my_matrix *MLh = matrix_alloc(N, part, .0);

  /* Main cycle of an iteration */
  for(int step = 0; step < N-1; step++)
  {
    /* Choose the task containing the column with the pivot
     * and calculating current values l_i */
    if(step % np == rank)
    {
      int col_index = (step - (step % np)) / np;
      MLh->data[step*part + col_index] = 1.;
      for(int i = step+1; i < N; i++)
      {
        MLh->data[i*part + col_index] = MAh->data[i*part + col_index] /
                                        MAh->data[step*part + col_index];
      }
      for(int i = 0; i < N; i++)
      {
        current_l->data[i] = MLh->data[i*part + col_index];
      }
    }
    /* Send current value of the vector l_i */
    MPI_Bcast(current_l, 1, vector_struct, step % np, MPI_COMM_WORLD);

    /* Modify the columns of the matrix MA
     * according to current value of the vector l */
    for(int i = step+1; i < N; i++)
    {
      for(int j = 0; j < part; j++)
      {
        MAh->data[i*part + j] -= MAh->data[step*part + j] * current_l->data[i];
      }
    }
  }

  /* Calculate the product of elements on the main diagonal
   * (with consideration of the column number of a task) */
  double prod = 1.;
  for(int i = 0; i < part; i++)
  {
    int row_index = i*np + rank;
    prod *= MAh->data[row_index*part + i];
  }

  /* Reduce the products of elements on the main diagonal
   * and output result in task 0 */
  if(rank == 0)
  {
    MPI_Reduce(MPI_IN_PLACE, &prod, 1, MPI_DOUBLE, MPI_PROD, 0, MPI_COMM_WORLD);
    printf("%lf", prod);
  }
  else
  {
    MPI_Reduce(&prod, NULL, 1, MPI_DOUBLE, MPI_PROD, 0, MPI_COMM_WORLD);
  }

  /* Free allocated resources */
  MPI_Type_free(&matrix_columns);
  MPI_Type_free(&vector_struct);
  return MPI_Finalize();
}

