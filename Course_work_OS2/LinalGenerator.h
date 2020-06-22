#pragma once
#ifndef LINAL_GENERATOR_H
#define LINAL_GENERATOR_H

#include <vector>
#include <random>

std::random_device rd;
std::mt19937 generator(rd());


// отрезок распределения -- опционален
const int a = 1; 
const int b = 100;


template<typename T>
std::vector<T> generateVector(const uint32_t n)
{
  std::vector<T> vec(n);
  std::uniform_int_distribution<int> uniDistribution(a, b);

  for (size_t i = 0;i < vec.size();++i) {
    vec[i] = uniDistribution(generator);
  }

  return vec;
}

template<typename T>
std::vector<std::vector<T>> generateGoodConditionedMatrix(const uint32_t n)
{
  std::vector<std::vector<T>> matrix(n, std::vector<T>(n));
  std::uniform_int_distribution<int> uniDistribution(a, b);

  for (size_t i = 0;i < matrix.size();++i) {
    int sum = 0;
    for (size_t j = 0;j < matrix[i].size();++j) {
      if (i != j) {
        matrix[i][j] = uniDistribution(generator);
        sum += matrix[i][j];
      }
    }
    if (sum < b) {
      uniDistribution.param(std::uniform_int_distribution<int>::param_type(b - sum + 1, b));
      matrix[i][i] = uniDistribution(generator);
      uniDistribution.param(std::uniform_int_distribution<int>::param_type(a, b));
    }
    else {
      matrix[i][i] = sum + 1;
    }
  }

  return matrix;
}

#endif // LINAL_GENERATOR_H