# Алгоритм метода «Внутренних точек» для решения задачи линейного программирования
## Текст задания

Решить задачу линейного программирования (минимизация функции при нескольких ограничениях), используя метод внутренних точек. 

- Ввод: данные читаются из файла формата .mps (не реализовано, ввод в консоль)
- Вывод: найденное решение - элементы вектора x и скаляр, с точностью до 6 знака после запятой.
## Вопросы со звездочкой

- можно ли инициализировать вектор x или параметр γ так, чтобы на большом числе задач, со
статистической значимостью, количество итераций алгоритма было меньше чем при случайной
инициализации вектора x или параметра γ

_[Перейти к примеру вычислений](#title1)_

- как на текущем шаге "Решаем СЛАУ относительно u" не решать СЛАУ с нуля, а использовать решение
СЛАУ с предыдущего шага для более быстрого получения решения на текущем шаге.

_Можно использовать [generalized minimal residual method](https://github.com/CristianCosci/GMRES?tab=readme-ov-file). В этом случае вычисления рекуррентны, а не считаются с 0 каждый раз, но данный способ сильно загружает память, из-за чего необходимо дополнительно оптимизировать затраты_

## Используемая библиотека

[Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) - библиотека C++ для линейной алгебры. Она предоставляет широкий спектр функций, включая работу с векторами, матрицами, разложениеми (SVD, QR, LU, Cholesky),  решение систем линейных уравнений, вычисление собственных значений и многое другое. 

## <a id="title1">Пример вычислений</a>

Изначальный вариант задания начального приближения - вектор единиц. Однако это не универсальный способ, который никак не упрощает вычисления, хотелось найти что-то лучше. Первой идеей было использовать обратную матрицу А, но возникла проблема - невозможность найти обратную матрицу для сингулярных матриц. Поэтому было решено использовать псевдообратную матрицу для задания начального приближения. На примере следующих входных данных сравним эти способы

Входные данные:
```
n, m = 3
A = [[1, 2, 6], [7, 0, 3], [6, 3, -1]]
b = [1, 1, 1]
c = [-5, 0, -3]
Y = 0.5
E = 10^-6
mex_depth = 120
Начальное приближение - вектор единиц
```
Результат работы программы:
```
Iterations: 1
Solution x:
0.503311 0.536424      0.5
C^T * x = -4.016556
Iterations: 2
Solution x:
0.254967 0.304636 0.250000
C^T * x = -2.024834
Iterations: 3
Solution x:
0.130795 0.188742 0.125000
C^T * x = -1.028974
Iterations: 4
Solution x:
0.101796 0.161677 0.095808
C^T * x = -0.796407
```
Входные данные:
```
n, m = 3
A = [[1, 2, 6], [7, 0, 3], [6, 3, -1]]
b = [1, 1, 1]
c = [-5, 0, -3]
Y = 0.5
E = 10^-6
mex_depth = 120
Начальное приближение - A^+b, где A^+ - псевдообратная матрица
//    x(0) = 0.101796;
//    x(1) = 0.161677;
//    x(2) = 0.095808;
```
Результат работы программы:
```
Iterations: 1
Solution x:
0.101796 0.161677 0.095808
C^T * x = -0.796407
```
Оба способа получают одинаковые итоговые значения с той разницей, что вариант с предварительно вычисленным значением начального приближения справился за 1 итерацию, а не за 4

Вычислим невязки с полученными значениями:


1 - (1*0.101796 + 2*0.161677 + 6*0.095808) = 0.000002


1 - (7*0.101796 + 0*0.161677 + 3*0.095808) = 0.000004


1 - (6*0.101796 + 3*0.161677 - 1*0.095808) = 0.000001


Результаты вычислений позволяют говорить, что решение найдено верно. 

