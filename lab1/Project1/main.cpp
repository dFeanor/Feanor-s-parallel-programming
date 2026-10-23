#include <omp.h>
#include <iostream>
#include <cmath>
using namespace std;

double f(double x)
{
    double y = x + 1.0;

    for (int k = 0; k < 10000000; k++)
    {
        y = std::sin(y)
            + std::cos(y * 0.5)
            + std::sqrt(std::abs(y) + 1.0);
    }
    return y;
}

int main()
{
    double a[100], b[100];

    for (int i = 0; i < 100; i++)
        b[i] = i;

 #pragma omp parallel for
    for (int i = 0; i < 100; i++)
    {
        a[i] = f(b[i]);
        b[i] = 2 * a[i];
    }

    double result = 0;

#pragma omp parallel for reduction(+ : result)
    for (int i = 0; i < 100; i++)
        result += a[i] + b[i];

    cout << "Result = " << result << endl;

    return 0;
}