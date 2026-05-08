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

constexpr int NUM_RUNS = 30;          // количество запусков
constexpr uint64_t BASE_SEED = 234356;
constexpr int dims = 10;
constexpr int steps = 20000;
const hrect rect(make_pair(-100.0, 100.0), dims);

template<double C>
double inverse(const hvector<dims>& v, double c) {
    const auto* x = v.data();
    double val = cec17_error(cec17_fitness(const_cast<double*>(x)));
    constexpr double coef = C * dims;
    return max(0.0, coef - val);
}

template<double C>
double inverseSigmoid(const hvector<dims>& v) {
    const auto* x = v.data();
    double val = cec17_error(cec17_fitness(const_cast<double*>(x)));
    constexpr double coef = C / dims;
    return 1.0 / (1.0 + exp(coef * val));
}

template<typename AlgoT>
void run_test(const string& algoName) {
    using FuncPtr = double (*)(const hvector<dims>&);
    const array<pair<int, FuncPtr>, 6> funcIds = { {
        { 5, inverseSigmoid<1e-3> },      // rastrigin
        { 1, inverseSigmoid<1e-8> },      // bent cigar
        { 16, inverseSigmoid<1e-2> },     // hybrid 6
        { 19, inverseSigmoid<1e-4> },     // hybrid 9
        { 28, inverseSigmoid<1e-8> },     // composition 8
        { 30, inverseSigmoid<1e-7> }      // composition 10
    } };

    cout << algoName << '\n';
    for (auto funcid : funcIds) {
        for (int run = 0; run < NUM_RUNS; ++run) {
            uint64_t seed = BASE_SEED + run * 1000 + funcid.first;
            cec17_init(algoName.c_str(), funcid.first, dims);

            AlgoT algo(funcid.second, rect, seed);
            auto sol = algo.result(steps, true);

            // Сохраняем каждый запуск в отдельный файл
            string filename = algoName + "_" + to_string(funcid.first) + "_run" + to_string(run) + ".csv";
            saveBestsToCSV(algo.dump_bests(), filename);
        }
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