#include <iostream>
#include <vector>
#include <random>
#include <array>
#include <filesystem>
#include <string>

#include "hvector.hpp"
#include "hrect.hpp"
#include "sofa_base.hpp"

extern "C" {
#include "cec17.h"
}

using namespace std;
using namespace swarm_algorithm;

#define PI 3.1415926535897932384626433832795029

void my_weierstrass_func(double* x, double* f, int nx) /* Weierstrass's  */
{
    int i, j, k_max;
    double sum, sum2, a, b;
    a = 0.5;
    b = 3.0;
    k_max = 20;
    f[0] = 0.0;

    for (i = 0; i < nx; i++) {
        sum = 0.0;
        sum2 = 0.0;
        for (j = 0; j <= k_max; j++) {
            sum += pow(a, j) * cos(2.0 * PI * pow(b, j) * (x[i] + 0.5));
            sum2 += pow(a, j) * cos(2.0 * PI * pow(b, j) * 0.5);
        }
        f[0] += sum;
    }
    f[0] -= nx * sum2;
}


void my_griewank_func(double* x, double* f, int nx) /* Griewank's  */
{
    int i;
    double s, p;
    s = 0.0;
    p = 1.0;

    for (i = 0; i < nx; i++) {
        s += x[i] * x[i];
        p *= cos(x[i] / sqrt(1.0 + i));
    }
    f[0] = 1.0 + s / 4000.0 - p;
}

constexpr int dims = 2;
constexpr int steps = 20000 - 100;
const string algo_name = "SOFA";

int main() {
    filesystem::create_directory("results_" + algo_name);


    const auto func = [](const hvector<dims>& v) -> double {
        
        constexpr double coef = 1.0 / 20.0;

        const auto* x = v.data();
        
        double val = cec17_error(cec17_fitness(const_cast<double*>(x)));
        /*double val = 0.0;
        my_weierstrass_func(const_cast<double*>(x), &val, dims);*/

        return 1.0 / (1.0 + std::exp(coef * val));
        };

    hrect rect(std::make_pair(-100.0, 100.0), dims);

    //array<int, 6> funcIds = { 5, 1, 16, 19, 28, 30 };
    array<int, 4> funcIds = { 5, 10, 1, 28};

    for (auto funcid : funcIds) {
        cec17_init(algo_name.c_str(), funcid, dims);

        sofa_base<dims> algo(func, rect, 234356);
        auto sol = algo.result(steps, true);

        cout << "Best Random[F" << funcid << "]: " << sol.second << endl;
        //for (int i = 0; i < sol.first.dims(); i++) cout << sol.first[i] << ' ';
        //cout << endl;

        auto bests = algo.dump_bests();
        for (auto& vv : bests) {
            cout << "Step: " << get<0>(vv) << " Value: " << get<2>(vv) << " Pos: \n";
            //for (int i = 0; i < get<1>(vv).dims(); i++) cout << get<1>(vv)[i] << ' ';
            //cout << '\n';
        }
    }

    return 0;
}