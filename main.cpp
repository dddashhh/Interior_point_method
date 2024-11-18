#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;


VectorXd solveCholesky(const MatrixXd& A, const VectorXd& b) {
    LLT<MatrixXd> llt(A);
    return llt.solve(b);
}

int main() {
    int n, m, max_depth;
    double tolerance, gamma;


    cout << "Введите размерность вектора x (n): ";
    cin >> n;
    cout << "Введите размерность вектора b (m): ";
    cin >> m;

    if (n <= 0 || m <= 0) {
        cerr << "Ошибка: Размеры должны быть положительными." << endl;
        return 1;
    }


    VectorXd c(n);
    cout << "Введите элементы вектора c (" << n << " чисел):" << endl;
    for (int i = 0; i < n; ++i) {
        cin >> c(i);
    }

    VectorXd b(m);
    cout << "Введите элементы вектора b (" << m << " чисел):" << endl;
    for (int i = 0; i < m; ++i) {
        cin >> b(i);
    }
    MatrixXd A(m, n);
    cout << "Введите элементы матрицы A (" << m << " строк, " << n << " столбцов):" << endl;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            cin >> A(i, j);
        }
    }


    cout << "Введите параметр gamma: ";
    cin >> gamma;
    cout << "Введите точность tolerance: ";
    cin >> tolerance;
    cout << "Введите максимальное число итераций max_depth: ";
    cin >> max_depth;

    if (tolerance <= 0 || max_depth <= 0) {
        cerr << "Ошибка: Tolerance и max_depth должны быть положительными." << endl;
        return 1;
    }
    VectorXd x = VectorXd::Ones(n);



    VectorXd r = b - A * x;
    int iteration = 0;
    while ((r.norm() >= tolerance) && (iteration < max_depth)) {
        iteration++;

        MatrixXd D = x.array().square().matrix().asDiagonal();
        MatrixXd T = A * D * A.transpose();
        VectorXd q = r + A * D * c;

        try {
            VectorXd u = solveCholesky(T, q);
            VectorXd g = c - A.transpose() * u;
            VectorXd s = -D * g;

            double lambda_hat = numeric_limits<double>::infinity();
            for (int j = 0; j < n; ++j) {
                if (s(j) < 0) {
                    lambda_hat = min(lambda_hat, -x(j) / s(j));
                }
            }

            double lambda_k = (r.norm() > 0) ? min(1.0, gamma * lambda_hat) : gamma * lambda_hat;
            x += lambda_k * s;
            r = b - A * x;
        } catch (const runtime_error& error) {
            cerr << "Error during iteration: " << error.what() << endl;
            return 1;
        }
    }

    
    cout << "Iterations: " << iteration << endl;
    cout << "Solution x:" << endl << x.transpose() << endl;
    cout << "C^T * x = " << fixed << setprecision(6) << c.transpose() * x << endl;

    return 0;
}
