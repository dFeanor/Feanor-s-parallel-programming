#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <chrono>

using namespace std;


// Генерация маленькой тестовой СЛАУ
void generateSmallSystem(vector<double>& A,
    vector<double>& b,
    vector<double>& x_exact)
{
    A = {
        10, -1,  2,  0,
        -1, 11, -1,  3,
         2, -1, 10, -1,
         0,  3, -1,  8
    };

    b = {
         6,
        25,
       -11,
        15
    };

    x_exact = {
         1,
         2,
        -1,
         1
    };
}


// Генерация большой СЛАУ
void generateLargeSystem(int N,
    vector<double>& A,
    vector<double>& b,
    vector<double>& x_exact)
{
    A.resize(N * N);
    b.resize(N);
    x_exact.resize(N);

    // Задаем точное решение
    for (int i = 0; i < N; i++)
    {
        x_exact[i] = 1.0 + 0.1 * (i % 10);
    }

    // Заполняем матрицу
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (i == j)
                A[i * N + j] = 2.0 * N;
            else
                A[i * N + j] = 1.0;
        }
    }

    // Вычисляем b = A * x_exact
    for (int i = 0; i < N; i++)
    {
        b[i] = 0.0;

        for (int j = 0; j < N; j++)
        {
            b[i] += A[i * N + j] * x_exact[j];
        }
    }
}

// Параллельный метод Якоби
void jacobiParallel(vector<double>& A,
    vector<double>& b,
    vector<double>& x,
    int N,
    double eps,
    int max_iterations,
    int& iterations)
{
    vector<double> x_new(N, 0.0);

    x.assign(N, 0.0);

    iterations = 0;

    while (iterations < max_iterations)
    {
        double residual = 0.0;

#pragma omp parallel for reduction(+:residual)

        for (int i = 0; i < N; i++)
        {
            double sum = 0.0;

            for (int j = 0; j < N; j++)
            {
                sum += A[i * N + j] * x[j];
            }

            double r = b[i] - sum;

            residual += r * r;

            x_new[i] = x[i] + r / A[i * N + i];
        }

        residual = sqrt(residual);

        if (residual < eps)
            break;

        x = x_new;

        iterations++;
    }
}


int main()
{
    double eps = 1e-8;

    int max_iterations = 10000;

    int N_large = 8000;

#ifdef _OPENMP
#pragma omp parallel
    {
#pragma omp single
        cout << "Number of threads: "
            << omp_get_num_threads() << endl;
    }
#else
    cout << "Number of threads: 1" << endl;
#endif

// Тест на небольшой системе
    int N_small = 4;

    vector<double> A_small;
    vector<double> b_small;
    vector<double> x_exact_small;
    vector<double> x_small;

    generateSmallSystem(
        A_small,
        b_small,
        x_exact_small
    );

    int iterations_small;

    cout << "Exact solution:" << endl;

    for (int i = 0; i < N_small; i++)
    {
        cout << x_exact_small[i] << " ";
    }
    cout << endl;

    auto start = chrono::high_resolution_clock::now();

    jacobiParallel(A_small, b_small, x_small, N_small, eps, max_iterations, iterations_small);

    auto end = chrono::high_resolution_clock::now();

    double time =
        chrono::duration<double>(end - start).count();

    cout << "Jacobi solution:" << endl;

    for (int i = 0; i < N_small; i++)
    {
        cout << x_small[i] << " ";
    }
    cout << endl;

    cout << "Iterations: " << iterations_small << endl;
    cout << "Time: " << time << endl;

// Тест на большой системе
    cout << "N = " << N_large << endl;

    vector<double> A;
    vector<double> b;
    vector<double> x_exact;
    vector<double> x;

    generateLargeSystem(
        N_large,
        A,
        b,
        x_exact
    );
 
    int iterations_large;

    cout << "Exact solution:" << endl;

    for (int i = 0; i < 13; i++)
    {
        cout << x_exact[i] << " ";
    }
    cout << " ..." << endl;
 
    start = chrono::high_resolution_clock::now();
 
    jacobiParallel(A,b,x,N_large,eps,max_iterations,iterations_large);
 
    end = chrono::high_resolution_clock::now();
    time = chrono::duration<double>(end - start).count();

    cout << "Jacobi solution:" << endl;

    for (int i = 0; i < 13; i++)
    {
        cout << x[i] << " ";
    }
    cout << " ..." << endl;
 
    cout << "Time: " << time << " s" << endl;
 
    cout << "Iterations: " << iterations_large << endl;
 
    return 0;
}