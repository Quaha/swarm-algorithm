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
#include "sofam.hpp"
#include "sofamg.hpp"
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
const hrect rect(make_pair(-100.0, 100.0), dims);

double inverse(const hvector<dims>& v) {
    const auto* x = v.data();
    double val = cec17_error(cec17_fitness(const_cast<double*>(x)));
    constexpr double coef = 1e4 * dims;
    return max(0.0, coef - val);
}

double inverseSigmoid(const hvector<dims>& v) {
    const auto* x = v.data();
    double val = cec17_error(cec17_fitness(const_cast<double*>(x)));
    constexpr double coef = 1e-10 / dims;
    return 1.0 / (1.0 + exp(coef * val));
}

template<typename AlgoT>
void run_test(const string& algoName) {
    using FuncPtr = double (*)(const hvector<dims>&);
    const array<pair<int, FuncPtr>, 6> funcIds = { {
        { 5, inverseSigmoid },      // rastrigin
        { 1, inverseSigmoid },      // bent cigar
        { 16, inverse },     // hybrid 6
        { 19, inverseSigmoid },     // hybrid 9
        { 28, inverse },     // composition 8
        { 30, inverseSigmoid }      // composition 10
    } };

    cout << algoName << '\n';
    for (auto funcid : funcIds) {
        cec17_init(algoName.c_str(), funcid.first, dims);

        AlgoT algo(funcid.second, rect, 234356);
        auto sol = algo.result(steps, true);

        cout << "Best "+algoName+"[F" << funcid.first << "]: " << sol.second << endl;

        saveBestsToCSV(algo.dump_bests(), algoName + "_" + to_string(funcid.first) + ".csv");
    }
}

int main() {

    run_test<sofa_base<dims>>("SoFA");
    run_test<sofam_base<dims>>("SoFAM");
    run_test<de_base<dims>>("DE");
    run_test<mingo_base<dims>>("MINGO");
    run_test<crs_base<dims>>("CRS");
    run_test<sofamg_base<dims>>("SoFAMG");

    return 0;
}