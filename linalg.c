#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#include <mpi.h>

#include "linalg.h"

void fatal_error(const char *message, int errorcode)
{
  printf("fatal error: code %d, %s\n", errorcode, message);
  fflush(stdout);
  MPI_Abort(MPI_COMM_WORLD, errorcode);
}

struct my_vector *vector_alloc(int size, double initial)
{
  struct my_vector *result = malloc(sizeof(struct my_vector) +
                                    (size-1) * sizeof(double));
  result->size = size;

  for(int i = 0; i < size; i++)
  {
    result->data[i] = initial;
  }

  return result;
}

void vector_print(FILE *f, struct my_vector *vec)
{
  for(int i = 0; i < vec->size; i++)
  {
    fprintf(f, "%.15lf ", vec->data[i]);
  }
  fprintf(f, "\n");
}

struct my_matrix *matrix_alloc(int rows, int cols, double initial)
{
  struct my_matrix *result = malloc(sizeof(struct my_matrix) +
                                    (rows * cols - 1) * sizeof(double));

  result->rows = rows;
  result->cols = cols;

  for(int i = 0; i < rows; i++)
  {
    for(int j = 0; j < cols; j++)
    {
      result->data[i * cols + j] = initial;
    }
  }

  return result;
}

void matrix_print(FILE *f, struct my_matrix *mat)
{
  for(int i = 0; i < mat->rows; i++)
  {
    for(int j = 0; j < mat->cols; j++)
    {
      fprintf(f, "%lf ", mat->data[i * mat->cols + j]);
    }
    fprintf(f, "\n");
  }
}

struct my_matrix *read_matrix(const char *filename)
{
  FILE *mat_file = fopen(filename, "r");
  if(mat_file == NULL)
  {
    fatal_error("can't open matrix file", 1);
  }

  int rows;
  int cols;
  fscanf(mat_file, "%d %d", &rows, &cols);

  struct my_matrix *result = matrix_alloc(rows, cols, 0.0);
  for(int i = 0; i < rows; i++)
  {
    for(int j = 0; j < cols; j++)
    {
      fscanf(mat_file, "%lf", &result->data[i * cols + j]);
    }
  }

  fclose(mat_file);
  return result;
}

struct my_vector *read_vector(const char *filename)
{
  FILE *vec_file = fopen(filename, "r");
  if(vec_file == NULL)
  {
    fatal_error("can't open vector file", 1);
  }

  int size;
  fscanf(vec_file, "%d", &size);

  struct my_vector *result = vector_alloc(size, 0.0);
  for(int i = 0; i < size; i++)
  {
    fscanf(vec_file, "%lf", &result->data[i]);
  }

  fclose(vec_file);
  return result;
}

void write_vector(const char *filename, struct my_vector *vec)
{
  FILE *vec_file = fopen(filename, "w");
  if(vec_file == NULL)
  {
    fatal_error("can't open vector file", 1);
  }

  vector_print(vec_file, vec);

  fclose(vec_file);
}

