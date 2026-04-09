#include <iostream>
#include <vector>
#include <random>
#include <array>

#include "hvector.hpp"
#include "hrect.hpp"
#include "sofa_base.hpp"

extern "C" {
#include "cec17.h"
}

using namespace std;
using namespace swarm_algorithm;

#define PI 3.1415926535897932384626433832795029

int main() {

    constexpr int dims = 10;

    const auto func = [](const hvector<dims>& v) -> double {
        
        constexpr double coef = 1.0 / 20.0;

        const auto* x = v.data();
        
        double val = cec17_error(cec17_fitness(const_cast<double*>(x)));

        return 1.0 / (1.0 + std::exp(coef * val));
        };

    hrect rect({
        std::make_pair(-100.0, 100.0),
        std::make_pair(-100.0, 100.0),
        std::make_pair(-100.0, 100.0),
        std::make_pair(-100.0, 100.0),
        std::make_pair(-100.0, 100.0),
        std::make_pair(-100.0, 100.0),
        std::make_pair(-100.0, 100.0),
        std::make_pair(-100.0, 100.0),
        std::make_pair(-100.0, 100.0),
        std::make_pair(-100.0, 100.0),
        });

    //array<int, 6> funcIds = { 5, 1, 16, 19, 28, 30 };
    array<int, 3> funcIds = { 5, 1, 28 };

    for (auto funcid : funcIds) {
        cec17_init("SoFA", funcid, dims);

        sofa_base<dims> algo(func, rect, 234356);
        auto sol = algo.result(99900, true);

        cout << "Best Random[F" << funcid << "]: " << sol.second << endl;
        for (int i = 0; i < sol.first.dims(); i++) cout << sol.first[i] << ' ';
        cout << endl;

        auto bests = algo.dump_bests();
        for (auto& vv : bests) {
            cout << "Step: " << get<0>(vv) << " Value: " << get<2>(vv) << " Pos: ";
            for (int i = 0; i < get<1>(vv).dims(); i++) cout << get<1>(vv)[i] << ' ';
            cout << '\n';
        }
    }

    return 0;
}