#ifndef _LINALG_H_
#define _LINALG_H_

struct my_vector
{
  int size;
  double data[1];
};

struct my_matrix
{
  int rows;
  int cols;
  double data[1];
};

void fatal_error(const char *message, int errorcode);

struct my_vector *vector_alloc(int size, double initial);
void vector_print(FILE *f, struct my_vector *vec);
void write_vector(const char *filename, struct my_vector *vec);
struct my_vector *read_vector(const char *filename);

struct my_matrix *matrix_alloc(int rows, int cols, double initial);
void matrix_print(FILE *f, struct my_matrix *mat);
struct my_matrix *read_matrix(const char *filename);

#endif

