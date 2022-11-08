#include <iostream>
#include "taco.h"
#include <ctime>
#include <vector>
#include <cassert>

using namespace taco;

#define TI (double)clock()/CLOCKS_PER_SEC

Format coo({compressed(ModeFormat::Property::NOT_UNIQUE), Singleton(ModeFormat::Property::UNIQUE)});
Format csr({Dense, Sparse(ModeFormat::Property::UNIQUE)});
// Format dcsr({Sparse(ModeFormat::Property::UNIQUE),Sparse(ModeFormat::Property::UNIQUE)});
Format csc({Dense, Sparse(ModeFormat::Property::UNIQUE)}, {1,0});
// Format dcsc({Sparse, Sparse}, {1,0});

int evaluate_csr_csc(taco_tensor_t *A, taco_tensor_t *B) {
  int A2_dimension = (int)(A->dimensions[1]);
  int* A2_pos = (int*)(A->indices[1][0]);
  int* A2_crd = (int*)(A->indices[1][1]);
  double* A_vals = (double*)(A->vals);
  int B1_dimension = (int)(B->dimensions[0]);
  int* B2_pos = (int*)(B->indices[1][0]);
  int* B2_crd = (int*)(B->indices[1][1]);
  double* B_vals = (double*)(B->vals);

  int32_t* A2_attr_nnz = 0;

  int32_t A2_attr_nnz_capacity = A2_dimension;
  A2_attr_nnz = (int32_t*)calloc(A2_attr_nnz_capacity, sizeof(int32_t));
  for (int32_t i = 0; i < B1_dimension; i++) {
    for (int32_t pB2 = B2_pos[i]; pB2 < B2_pos[(i + 1)]; pB2++) {
      int32_t j = B2_crd[pB2];
      A2_attr_nnz[j] = A2_attr_nnz[j] + (int32_t)1;
    }
  }
  A2_pos = (int32_t*)malloc(sizeof(int32_t) * (A2_dimension + 1));
  A2_pos[0] = 0;
  for (int32_t jA = 0; jA < A2_dimension; jA++) {
    A2_pos[jA + 1] = A2_pos[jA] + A2_attr_nnz[jA];
  }
  A2_crd = (int32_t*)malloc(sizeof(int32_t) * A2_pos[A2_dimension]);
  int32_t A_capacity = A2_pos[A2_dimension];
  A_vals = (double*)malloc(sizeof(double) * A_capacity);
  for (int32_t i = 0; i < B1_dimension; i++) {
    for (int32_t pB2 = B2_pos[i]; pB2 < B2_pos[(i + 1)]; pB2++) {
      int32_t j = B2_crd[pB2];
      int32_t pA2 = A2_pos[j];
      A2_pos[j] = A2_pos[j] + 1;
      A2_crd[pA2] = i;
      A_vals[pA2] = B_vals[pB2];
    }
  }

  free(A2_attr_nnz);
  for (int32_t p = 0; p < A2_dimension; p++) {
    A2_pos[A2_dimension - p] = A2_pos[(A2_dimension - p - 1)];
  }
  A2_pos[0] = 0;

  A->indices[1][0] = (uint8_t*)(A2_pos);
  A->indices[1][1] = (uint8_t*)(A2_crd);
  A->vals = (uint8_t*)A_vals;
  return 0;
}

int evaluate_csr_unsorted_dia(taco_tensor_t *A, taco_tensor_t *B) {
  int A1_dimension = (int)(A->dimensions[0]);
  int A2_dimension = (int)(A->dimensions[1]);
  int* A1_perm = (int*)(A->indices[0][0]);
  int* A1_nzslice = (int*)(A->indices[0][1]);
  int* A1_shift = (int*)(A->indices[0][2]);

  double* A_vals = (double*)(A->vals);
  
  int B1_dimension = (int)(B->dimensions[0]);
  int* B2_pos = (int*)(B->indices[1][0]);
  int* B2_crd = (int*)(B->indices[1][1]);
  double* B_vals = (double*)(B->vals);

  bool* A1_attr_nonempty = 0;

  static std::vector<int> index;
  index.clear();
  index.resize(B1_dimension + B->dimensions[1], 0);
  std::vector<double*> new_val;
  std::vector<uint8_t> new_crd;

  int k = 0;
  for (int32_t i = 0; i < B1_dimension; i++) {
    for (int32_t pB2 = B2_pos[i]; pB2 < B2_pos[(i + 1)]; pB2++) {
      int32_t crd_1 = B2_crd[pB2] - i + B1_dimension;
      int32_t crd_2 = i;
      bool mark = 0;
      if (index[crd_1] == 0) {
        index[crd_1] = (k++);
        double* new_array = (double*)calloc(B1_dimension, sizeof(double));
        new_val.push_back(new_array);
        mark = 1;
      }
      new_val[index[crd_1]][crd_2] = B_vals[pB2];
      if (mark) new_crd.push_back(crd_1);
    }
  }

  A1_nzslice = (int32_t*)malloc(sizeof(int32_t) * 1);
  A1_nzslice[0] = new_crd.size();

  A->indices[0][0] = (uint8_t*)(new_crd.data());
  A->indices[0][1] = (uint8_t*)(A1_nzslice);
  A->indices[0][2] = (uint8_t*)(A1_shift);
  // A->vals = (uint8_t*)A_vals;
  return 0;
}

int evaluate_csr_dia(taco_tensor_t *A, taco_tensor_t *B) {
  int A1_dimension = (int)(A->dimensions[0]);
  int A2_dimension = (int)(A->dimensions[1]);
  int* A1_perm = (int*)(A->indices[0][0]);
  int* A1_nzslice = (int*)(A->indices[0][1]);
  int* A1_shift = (int*)(A->indices[0][2]);
  

  double* A_vals = (double*)(A->vals);
  
  int B1_dimension = (int)(B->dimensions[0]);
  int* B2_pos = (int*)(B->indices[1][0]);
  int* B2_crd = (int*)(B->indices[1][1]);
  double* B_vals = (double*)(B->vals);

  

  bool* A1_attr_nonempty = 0;

  int32_t A1_attr_nonempty_capacity = A1_dimension;
  A1_attr_nonempty = (bool*)calloc(A1_attr_nonempty_capacity, sizeof(bool));
  for (int32_t i = 0; i < B1_dimension; i++) {
    for (int32_t pB2 = B2_pos[i]; pB2 < B2_pos[(i + 1)]; pB2++) {
      int32_t j = B2_crd[pB2];
      int32_t i139A1_attr_nonempty = j - i + B1_dimension;
      A1_attr_nonempty[i139A1_attr_nonempty] = 1;
    }
  }

  
  A1_perm = (int32_t*)malloc(sizeof(int32_t) * A1_dimension);
  A1_nzslice = (int32_t*)malloc(sizeof(int32_t) * 1);
  A1_nzslice[0] = 0;
  for (int32_t i0 = 0; i0 < A1_dimension; i0++) {
    if (A1_attr_nonempty[i0] == 1) {
      A1_perm[A1_nzslice[0]] = i0;
      A1_nzslice[0] = A1_nzslice[0] + 1;
    }
  }
  
  int32_t* A1_rperm = 0;
  A1_rperm = (int32_t*)malloc(sizeof(int32_t) * A1_dimension);
  int32_t A1_nzslice_local = A1_nzslice[0];
  for (int32_t p = 0; p < A1_nzslice_local; p++) {
    A1_rperm[A1_perm[p]] = p;
  }
  std::cerr << A1_nzslice[0] << ' ' << A2_dimension << std::endl;
  if (A1_nzslice[0] > 1e5 && A2_dimension > 1e5) {
    std::cerr << "Warning: conversion must failed due to large index" << std::endl;
  }
  int32_t A_capacity = A1_nzslice[0] * A2_dimension;
  A_vals = (double*)calloc(A_capacity, sizeof(double));
  for (int32_t i = 0; i < B1_dimension; i++) {
    for (int32_t pB2 = B2_pos[i]; pB2 < B2_pos[(i + 1)]; pB2++) {
      int32_t j = B2_crd[pB2];
      int32_t i150A = j - i + B1_dimension;
      int32_t pA1 = A1_rperm[i150A];
      int32_t pA2 = pA1 * A2_dimension + i;
      assert(pA2 < A_capacity);
      A_vals[pA2] = B_vals[pB2];
    }
  }

  free(A1_attr_nonempty);
  free(A1_rperm);

  A->indices[0][0] = (uint8_t*)(A1_perm);
  A->indices[0][1] = (uint8_t*)(A1_nzslice);
  A->indices[0][2] = (uint8_t*)(A1_shift);
  A->vals = (uint8_t*)A_vals;
  return 0;
}

int evaluate_coo_dia(taco_tensor_t *A, taco_tensor_t *B) {
  int A1_dimension = (int)(A->dimensions[0]);
  int A2_dimension = (int)(A->dimensions[1]);
  int* A1_perm = (int*)(A->indices[0][0]);
  int* A1_nzslice = (int*)(A->indices[0][1]);
  int* A1_shift = (int*)(A->indices[0][2]);
  double* A_vals = (double*)(A->vals);
  int* B1_pos = (int*)(B->indices[0][0]);
  int* B1_crd = (int*)(B->indices[0][1]);
  int* B2_crd = (int*)(B->indices[1][1]);
  int B1_dimension = (int)(B->dimensions[0]);
  double* B_vals = (double*)(B->vals);

  bool* A1_attr_nonempty = 0;

  int32_t A1_attr_nonempty_capacity = A1_dimension;
  A1_attr_nonempty = (bool*)calloc(A1_attr_nonempty_capacity, sizeof(bool));
  for (int32_t pB1 = B1_pos[0]; pB1 < B1_pos[1]; pB1++) {
    int32_t i = B1_crd[pB1];
    int32_t j = B2_crd[pB1];
    int32_t i139A1_attr_nonempty = j - i + B1_dimension;
    A1_attr_nonempty[i139A1_attr_nonempty] = 1;
  }
  A1_perm = (int32_t*)malloc(sizeof(int32_t) * A1_dimension);
  A1_nzslice = (int32_t*)malloc(sizeof(int32_t) * 1);
  A1_nzslice[0] = 0;
  for (int32_t i0 = 0; i0 < A1_dimension; i0++) {
    if (A1_attr_nonempty[i0] == 1) {
      A1_perm[A1_nzslice[0]] = i0;
      A1_nzslice[0] = A1_nzslice[0] + 1;
    }
  }
  int32_t* A1_rperm = 0;
  A1_rperm = (int32_t*)malloc(sizeof(int32_t) * A1_dimension);
  int32_t A1_nzslice_local = A1_nzslice[0];
  for (int32_t p = 0; p < A1_nzslice_local; p++) {
    A1_rperm[A1_perm[p]] = p;
  }
  std::cerr << A1_nzslice[0] << ' ' << A2_dimension << std::endl;
  if (A1_nzslice[0] > 1e5 && A2_dimension > 1e5) {
    std::cerr << "Warning: conversion must failed due to large index" << std::endl;
  }
  int32_t A_capacity = A1_nzslice[0] * A2_dimension;
  A_vals = (double*)calloc(A_capacity, sizeof(double));
  for (int32_t pB1 = B1_pos[0]; pB1 < B1_pos[1]; pB1++) {
    int32_t i = B1_crd[pB1];
    int32_t j = B2_crd[pB1];
    int32_t i150A = j - i + B1_dimension;
    int32_t pA1 = A1_rperm[i150A];
    int32_t pA2 = pA1 * A2_dimension + i;
    A_vals[pA2] = B_vals[pB1];
  }

  free(A1_attr_nonempty);
  free(A1_rperm);

  A->indices[0][0] = (uint8_t*)(A1_perm);
  A->indices[0][1] = (uint8_t*)(A1_nzslice);
  A->indices[0][2] = (uint8_t*)(A1_shift);
  A->vals = (uint8_t*)A_vals;
  return 0;
}

int evaluate_coo_csr(taco_tensor_t *A, taco_tensor_t *B) {
  int A1_dimension = (int)(A->dimensions[0]);
  int* A2_pos = (int*)(A->indices[1][0]);
  int* A2_crd = (int*)(A->indices[1][1]);
  double* A_vals = (double*)(A->vals);
  int* B1_pos = (int*)(B->indices[0][0]);
  int* B1_crd = (int*)(B->indices[0][1]);
  int* B2_crd = (int*)(B->indices[1][1]);
  double* B_vals = (double*)(B->vals);

  int32_t* A2_attr_nnz = 0;

  int32_t A2_attr_nnz_capacity = A1_dimension;
  A2_attr_nnz = (int32_t*)calloc(A2_attr_nnz_capacity, sizeof(int32_t));
  for (int32_t pB1 = B1_pos[0]; pB1 < B1_pos[1]; pB1++) {
    int32_t i = B1_crd[pB1];
    A2_attr_nnz[i] = A2_attr_nnz[i] + (int32_t)1;
  }
  A2_pos = (int32_t*)malloc(sizeof(int32_t) * (A1_dimension + 1));
  A2_pos[0] = 0;
  for (int32_t iA = 0; iA < A1_dimension; iA++) {
    A2_pos[iA + 1] = A2_pos[iA] + A2_attr_nnz[iA];
  }
  A2_crd = (int32_t*)malloc(sizeof(int32_t) * A2_pos[A1_dimension]);
  int32_t A_capacity = A2_pos[A1_dimension];
  A_vals = (double*)malloc(sizeof(double) * A_capacity);
  for (int32_t pB1 = B1_pos[0]; pB1 < B1_pos[1]; pB1++) {
    int32_t i = B1_crd[pB1];
    int32_t j = B2_crd[pB1];
    int32_t pA2 = A2_pos[i];
    A2_pos[i] = A2_pos[i] + 1;
    A2_crd[pA2] = j;
    A_vals[pA2] = B_vals[pB1];
  }

  free(A2_attr_nnz);
  for (int32_t p = 0; p < A1_dimension; p++) {
    A2_pos[A1_dimension - p] = A2_pos[(A1_dimension - p - 1)];
  }
  A2_pos[0] = 0;

  A->indices[1][0] = (uint8_t*)(A2_pos);
  A->indices[1][1] = (uint8_t*)(A2_crd);
  A->vals = (uint8_t*)A_vals;
  return 0;
}

int evaluate_csc_dia(taco_tensor_t *A, taco_tensor_t *B) {
  int A1_dimension = (int)(A->dimensions[0]);
  int A2_dimension = (int)(A->dimensions[1]);
  int* A1_perm = (int*)(A->indices[0][0]);
  int* A1_nzslice = (int*)(A->indices[0][1]);
  int* A1_shift = (int*)(A->indices[0][2]);
  double* A_vals = (double*)(A->vals);
  int B2_dimension = (int)(B->dimensions[1]);
  int* B2_pos = (int*)(B->indices[1][0]);
  int* B2_crd = (int*)(B->indices[1][1]);
  double* B_vals = (double*)(B->vals);

  bool* A1_attr_nonempty = 0;

  int32_t A1_attr_nonempty_capacity = A1_dimension;
  A1_attr_nonempty = (bool*)calloc(A1_attr_nonempty_capacity, sizeof(bool));
  for (int32_t j = 0; j < B2_dimension; j++) {
    for (int32_t pB2 = B2_pos[j]; pB2 < B2_pos[(j + 1)]; pB2++) {
      int32_t i = B2_crd[pB2];
      int32_t i139A1_attr_nonempty = j - i + B2_dimension;
      A1_attr_nonempty[i139A1_attr_nonempty] = 1;
    }
  }
  A1_perm = (int32_t*)malloc(sizeof(int32_t) * A1_dimension);
  A1_nzslice = (int32_t*)malloc(sizeof(int32_t) * 1);
  A1_nzslice[0] = 0;
  for (int32_t i0 = 0; i0 < A1_dimension; i0++) {
    if (A1_attr_nonempty[i0] == 1) {
      A1_perm[A1_nzslice[0]] = i0;
      A1_nzslice[0] = A1_nzslice[0] + 1;
    }
  }
  int32_t* A1_rperm = 0;
  A1_rperm = (int32_t*)malloc(sizeof(int32_t) * A1_dimension);
  int32_t A1_nzslice_local = A1_nzslice[0];
  for (int32_t p = 0; p < A1_nzslice_local; p++) {
    A1_rperm[A1_perm[p]] = p;
  }
  std::cerr << A1_nzslice[0] << ' ' << A2_dimension << std::endl;
  if (A1_nzslice[0] > 1e5 && A2_dimension > 1e5) {
    std::cerr << "Warning: conversion must failed due to large index" << std::endl;
  }
  int32_t A_capacity = A1_nzslice[0] * A2_dimension;
  A_vals = (double*)calloc(A_capacity, sizeof(double));
  for (int32_t j = 0; j < B2_dimension; j++) {
    for (int32_t pB2 = B2_pos[j]; pB2 < B2_pos[(j + 1)]; pB2++) {
      int32_t i = B2_crd[pB2];
      int32_t i150A = j - i + B2_dimension;
      int32_t pA1 = A1_rperm[i150A];
      int32_t pA2 = pA1 * A2_dimension + i;
      A_vals[pA2] = B_vals[pB2];
    }
  }

  free(A1_attr_nonempty);
  free(A1_rperm);

  A->indices[0][0] = (uint8_t*)(A1_perm);
  A->indices[0][1] = (uint8_t*)(A1_nzslice);
  A->indices[0][2] = (uint8_t*)(A1_shift);
  A->vals = (uint8_t*)A_vals;
  return 0;
}

int a[(const int)5e7] = {0};

void clear_cache() {
  memset(a, 1, sizeof(a));
}

float test_csr_dia(char* file_name) {
  Tensor<double> I = taco::read(file_name, csr);
  taco_tensor_t* B = I.getTacoTensorT();
  std::cerr << B->dimensions[0] << ' ' << B->dimensions[1] << std::endl;
  std::vector<int> bufferDim = {B->dimensions[1]+B->dimensions[0], B->dimensions[0], B->dimensions[1]};
  std::vector<int> bufferModeOrdering = {0, 1, 2}; //should not matter
  std::vector<taco_mode_t> bufferModeType = {taco_mode_dia1, taco_mode_sparse, taco_mode_sparse};
  taco_tensor_t* A = init_taco_tensor_t(3, 8, bufferDim.data(), bufferModeOrdering.data(), bufferModeType.data(), nullptr);
  float acc_time = 0;
  for (int i = 0; i < 50; ++i) {
    clear_cache();
    auto tic = TI;
    if (!evaluate_csr_dia(A, B)) {
      auto toc = TI;
      acc_time += toc-tic;
      deinit_taco_tensor_t(A);
      A = init_taco_tensor_t(3, 8, bufferDim.data(), bufferModeOrdering.data(), bufferModeType.data(), nullptr);
    }
  }
  return acc_time;
}

float test_csr_unsorted_dia(char* file_name) {
  Tensor<double> I = taco::read(file_name, csr);
  taco_tensor_t* B = I.getTacoTensorT();
  std::cerr << B->dimensions[0] << ' ' << B->dimensions[1] << std::endl;
  std::vector<int> bufferDim = {B->dimensions[1]+B->dimensions[0], B->dimensions[0], B->dimensions[1]};
  std::vector<int> bufferModeOrdering = {0, 1, 2}; //should not matter
  std::vector<taco_mode_t> bufferModeType = {taco_mode_dia1, taco_mode_sparse, taco_mode_sparse};
  taco_tensor_t* A = init_taco_tensor_t(3, 8, bufferDim.data(), bufferModeOrdering.data(), bufferModeType.data(), nullptr);
  float acc_time = 0;
  for (int i = 0; i < 50; ++i) {
    clear_cache();
    auto tic = TI;
    if (!evaluate_csr_unsorted_dia(A, B)) {
      auto toc = TI;
      acc_time += toc-tic;
      deinit_taco_tensor_t(A);
      A = init_taco_tensor_t(3, 8, bufferDim.data(), bufferModeOrdering.data(), bufferModeType.data(), nullptr);
    }
  }
  return acc_time;
}

float test_csc_dia(char* file_name) {
  Tensor<double> I = taco::read(file_name, csc);
  taco_tensor_t* B = I.getTacoTensorT();
  std::cerr << B->dimensions[0] << ' ' << B->dimensions[1] << std::endl;
  std::vector<int> bufferDim = {B->dimensions[1]+B->dimensions[0], B->dimensions[0], B->dimensions[1]};
  std::vector<int> bufferModeOrdering = {0, 1, 2}; //should not matter
  std::vector<taco_mode_t> bufferModeType = {taco_mode_dia1, taco_mode_sparse, taco_mode_sparse};
  taco_tensor_t* A = init_taco_tensor_t(3, 8, bufferDim.data(), bufferModeOrdering.data(), bufferModeType.data(), nullptr);
  float acc_time = 0;
  for (int i = 0; i < 50; ++i) {
    clear_cache();
    auto tic = TI;
    if (!evaluate_csr_dia(A, B)) {
      auto toc = TI;
      acc_time += toc-tic;
      deinit_taco_tensor_t(A);
      A = init_taco_tensor_t(3, 8, bufferDim.data(), bufferModeOrdering.data(), bufferModeType.data(), nullptr);
    }
  }
  return acc_time;
}

float test_csr_csc(char* file_name) {
  Tensor<double> I = taco::read(file_name, csr);
  taco_tensor_t* B = I.getTacoTensorT();
  std::cerr << B->dimensions[0] << ' ' << B->dimensions[1] << std::endl;
  std::vector<int> bufferDim = {B->dimensions[0], B->dimensions[1]};
  std::vector<int> bufferModeOrdering = {1, 0}; //should not matter
  std::vector<taco_mode_t> bufferModeType = {taco_mode_dense, taco_mode_sparse};
  taco_tensor_t* A = init_taco_tensor_t(2, 8, bufferDim.data(), bufferModeOrdering.data(), bufferModeType.data(), nullptr);
  float acc_time = 0;
  for (int i = 0; i < 50; ++i) {
    clear_cache();
    auto tic = TI;
    if (!evaluate_csr_csc(A, B)) {
      auto toc = TI;
      acc_time += toc-tic;
      deinit_taco_tensor_t(A);
      A = init_taco_tensor_t(2, 8, bufferDim.data(), bufferModeOrdering.data(), bufferModeType.data(), nullptr);
    }
  }
  return acc_time;
}

float test_coo_csr(char* file_name) {
  Tensor<double> I = taco::read(file_name, coo);
  taco_tensor_t* B = I.getTacoTensorT();
  std::cerr << B->dimensions[0] << ' ' << B->dimensions[1] << std::endl;
  std::vector<int> bufferDim = {B->dimensions[0], B->dimensions[1]};
  std::vector<int> bufferModeOrdering = {0, 1}; //should not matter
  std::vector<taco_mode_t> bufferModeType = {taco_mode_dense, taco_mode_sparse};
  taco_tensor_t* A = init_taco_tensor_t(2, 8, bufferDim.data(), bufferModeOrdering.data(), bufferModeType.data(), nullptr);
  float acc_time = 0;
  for (int i = 0; i < 50; ++i) {
    clear_cache();
    auto tic = TI;
    if (!evaluate_coo_csr(A, B)) {
      auto toc = TI;
      acc_time += toc-tic;
      deinit_taco_tensor_t(A);
      A = init_taco_tensor_t(2, 8, bufferDim.data(), bufferModeOrdering.data(), bufferModeType.data(), nullptr);
    }
  }
  return acc_time;
}

float test_coo_dia(char* file_name) {
  Tensor<double> I = taco::read(file_name, coo);
  taco_tensor_t* B = I.getTacoTensorT();
  std::cerr << B->dimensions[0] << ' ' << B->dimensions[1] << std::endl;
  std::vector<int> bufferDim = {B->dimensions[1]+B->dimensions[0], B->dimensions[0], B->dimensions[1]};
  std::vector<int> bufferModeOrdering = {0, 1, 2}; //should not matter
  std::vector<taco_mode_t> bufferModeType = {taco_mode_dia1, taco_mode_sparse, taco_mode_sparse};
  taco_tensor_t* A = init_taco_tensor_t(3, 8, bufferDim.data(), bufferModeOrdering.data(), bufferModeType.data(), nullptr);
  float acc_time = 0;
  for (int i = 0; i < 50; ++i) {
    clear_cache();
    auto tic = TI;
    if (!evaluate_coo_dia(A, B)) {
      auto toc = TI;
      acc_time += toc-tic;
      deinit_taco_tensor_t(A);
      A = init_taco_tensor_t(3, 8, bufferDim.data(), bufferModeOrdering.data(), bufferModeType.data(), nullptr);
    }
  }
  return acc_time;
}

int main(int argc, char* argv[]) {
  // Create formats

  if (argc < 3) {
    std::cerr << "[Usage] :: ./conversion tensor_file_name #test_case" << std::endl;
    return 0;
  }
  int test_case = atoi(argv[2]);
  float acc_time = 0;
  //coo_csr
  if (test_case == 0) {
    acc_time = test_coo_csr(argv[1]);
  } else if (test_case == 1) {
    acc_time = test_coo_dia(argv[1]);
  } else if (test_case == 2) {
    acc_time = test_csr_csc(argv[1]);
  } else if (test_case == 3) {
    acc_time = test_csr_dia(argv[1]);
  } else if (test_case == 4) {
    acc_time = test_csc_dia(argv[1]);
  } else if (test_case == 5) {
    acc_time = test_csr_unsorted_dia(argv[1]);
  }
  std::cerr << "average among 50 runs = " << ((float)acc_time*20.0) << "(ms)" << std::endl;
  return 0;
}