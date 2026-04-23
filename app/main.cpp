#include <iostream>
#include <vector>
#include <random>
#include <array>
#include <filesystem>
#include <string>

#include "printer.hpp"
#include "hvector.hpp"
#include "hrect.hpp"
#include "sofa_base.hpp"
#include "crs.hpp"
#include "mingo.hpp"
#include "de.hpp"

extern "C" {
#include "cec17.h"
}

using namespace std;
using namespace swarm_algorithm;

constexpr int dims = 10;
constexpr int steps = 20000;
hrect rect(std::make_pair(-100.0, 100.0), dims);

const auto inverse = [](const hvector<dims>& v) -> double {
    const auto* x = v.data();

    double val = cec17_error(cec17_fitness(const_cast<double*>(x)));

    return 20000.0 - val;
};

template<typename AlgoT>
void run_test(const string& algoName) {
    const static array<int, 5> funcIds = {
        5, // rastrigin
        1, // bent sigar
        16,// hybrid 6
        19,// hybrid 9
        28,// composition 8
        //30 // composition 10
    };

    cout << algoName << '\n';
    for (auto funcid : funcIds) {
        cec17_init(algoName.c_str(), funcid, dims);

        AlgoT algo(inverse, rect, 234356);
        auto sol = algo.result(steps, true);

        cout << "Best "+algoName+"[F" << funcid << "]: " << sol.second << endl;

        saveBestsToCSV(algo.dump_bests(), algoName + "_" + to_string(funcid) + ".csv");
    }
}

int main() {

    run_test<sofa_base<dims>>("SoFA");
    run_test<de_base<dims>>("DE");
    run_test<mingo_base<dims>>("MINGO");
    run_test<crs_base<dims>>("CRS");

    return 0;
}