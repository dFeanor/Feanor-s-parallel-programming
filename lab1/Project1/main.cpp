#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <chrono>

using namespace std;

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

void jacobiSequential(vector<double>& A,
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

void solveLargeSystemParallel(
    int N,
    double eps,
    int max_iterations,
    double*& x_exact,
    double*& x,
    int& iterations,
    double& calculation_time)
{
    double* A = new double[N * N];
    double* b = new double[N];

    x_exact = new double[N];

    x = new double[N];
    double* x_new = new double[N];

    double* current = x;
    double* next = x_new;

    double residual = 0.0;

    bool finished = false;

    chrono::high_resolution_clock::time_point start;
    chrono::high_resolution_clock::time_point end;

#pragma omp parallel
    {
#pragma omp single
        {
#ifdef _OPENMP
            cout << "Number of threads: "
                << omp_get_num_threads() << endl;
#else
            cout << "Number of threads: 1" << endl;
#endif
        }


        // ----------------------------------------------------
        // 1. Инициализация точного решения
        // ----------------------------------------------------

#pragma omp for schedule(static)
        for (int i = 0; i < N; i++)
        {
            x_exact[i] = 1.0 + 0.1 * (i % 10);
        }


        // ----------------------------------------------------
        // 2. Инициализация матрицы
        //
        // При schedule(static) строки распределяются
        // статически между потоками.
        // ----------------------------------------------------

#pragma omp for schedule(static)
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


        // ----------------------------------------------------
        // 3. Вычисляем b = A * x_exact
        //
        // Используем то же статическое распределение строк.
        // ----------------------------------------------------

#pragma omp for schedule(static)
        for (int i = 0; i < N; i++)
        {
            double sum = 0.0;

            for (int j = 0; j < N; j++)
            {
                sum += A[i * N + j] * x_exact[j];
            }

            b[i] = sum;
        }


        // ----------------------------------------------------
        // 4. Начальное приближение метода Якоби
        // ----------------------------------------------------

#pragma omp for schedule(static)
        for (int i = 0; i < N; i++)
        {
            current[i] = 0.0;
            next[i] = 0.0;
        }


        // ----------------------------------------------------
        // Начинаем измерять только время метода Якоби.
        //
        // Генерация A и b в это время не входит.
        // ----------------------------------------------------

#pragma omp single
        {
            iterations = 0;
            finished = false;
            residual = 0.0;
            start = chrono::high_resolution_clock::now();
        }

        while (true)
        {
#pragma omp for schedule(static) reduction(+:residual)
            for (int i = 0; i < N; i++)
            {
                double sum = 0.0;

                for (int j = 0; j < N; j++)
                {
                    sum += A[i * N + j] * current[j];
                }

                double r = b[i] - sum;

                residual += r * r;

                next[i] =
                    current[i]
                    + r / A[i * N + i];
            }


            // ------------------------------------------------
            // После omp for имеется неявный barrier.
            //
            // Поэтому здесь уже:
            // 1. посчитаны все next[i];
            // 2. полностью собран residual.
            // ------------------------------------------------

#pragma omp single
            {
                residual = sqrt(residual);

                if (residual < eps)
                {
                    finished = true;

                    end =
                        chrono::high_resolution_clock::now();
                }
                else
                {
                    double* temp = current;
                    current = next;
                    next = temp;

                    iterations++;


                    if (iterations >= max_iterations)
                    {
                        finished = true;

                        end =
                            chrono::high_resolution_clock::now();
                    }
                    else
                    {
                        residual = 0.0;
                    }
                }
            }

            if (finished)
                break;
        }

    }

    calculation_time =
        chrono::duration<double>(end - start).count();


    // current указывает на окончательное решение.
    x = current;

    delete[] next;

    delete[] A;
    delete[] b;
}

int main()
{
    double eps = 1e-8;

    int max_iterations = 10000;

    int N_large = 8000;


    // ========================================================
    // Тест на небольшой системе
    // ========================================================

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


    cout << "Small system" << endl;

    cout << "Exact solution:" << endl;

    for (int i = 0; i < N_small; i++)
    {
        cout << x_exact_small[i] << " ";
    }

    cout << endl;


    auto start =
        chrono::high_resolution_clock::now();


    jacobiSequential(
        A_small,
        b_small,
        x_small,
        N_small,
        eps,
        max_iterations,
        iterations_small
    );


    auto end =
        chrono::high_resolution_clock::now();


    double time_small =
        chrono::duration<double>(
            end - start
        ).count();


    cout << "Jacobi solution:" << endl;

    for (int i = 0; i < N_small; i++)
    {
        cout << x_small[i] << " ";
    }

    cout << endl;

    cout << "Iterations: "
        << iterations_small << endl;

    cout << "Time: "
        << time_small << " s" << endl;


    // ========================================================
    // Большая система
    // ========================================================

    cout << endl;

    cout << "Large system" << endl;

    cout << "N = "
        << N_large << endl;


    double* x_exact = nullptr;
    double* x = nullptr;

    int iterations_large;

    double calculation_time;


    solveLargeSystemParallel(
        N_large,
        eps,
        max_iterations,
        x_exact,
        x,
        iterations_large,
        calculation_time
    );


    cout << "Exact solution:" << endl;

    for (int i = 0; i < 13; i++)
    {
        cout << x_exact[i] << " ";
    }

    cout << "..." << endl;


    cout << "Jacobi solution:" << endl;

    for (int i = 0; i < 13; i++)
    {
        cout << x[i] << " ";
    }

    cout << "..." << endl;


    cout << "Time: "
        << calculation_time
        << " s" << endl;

    cout << "Iterations: "
        << iterations_large
        << endl;


    delete[] x_exact;
    delete[] x;


    return 0;
}