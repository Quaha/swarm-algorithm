#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include "hvector.hpp"
#include "mingo.hpp"

using namespace swarm_algorithm;

// MINGO combines four stochastic strategies (Levy flight, Brownian
// motion, Cauchy mutation, and a reinforced exploitation rule). It is
// more robust than the base NGO but, like any metaheuristic, is not
// guaranteed to find the global optimum on every seed. The tests below
// exercise:
//   1. SphereConvergence                 -- deterministic-ish (unimodal)
//   2. RastriginFindsGlobalOptimum       -- one known-good seed
//   3. RastriginGlobalOptimumMajorityRun -- multi-seed statistical check

TEST(MingoBaseTests, SphereConvergence) {

    // Unimodal, global max at (1, 1), f* = 1.
    const auto func = [](const hvector<2>& v) -> double {
        double x = v[0] - 1.0;
        double y = v[1] - 1.0;
        return 1.0 / (1.0 + x * x + y * y);
        };

    hrect rect({
        std::make_pair(-3.0, 3.0),
        std::make_pair(-3.0, 3.0)
        });

    mingo_base<2> algo(func, rect, 23445);
    auto res = algo.result(10000);

    EXPECT_LE(std::abs(res.first[0] - 1.0), 0.01);
    EXPECT_LE(std::abs(res.first[1] - 1.0), 0.01);
    EXPECT_GE(res.second, 0.9999);
}

TEST(MingoBaseTests, RastriginFindsGlobalOptimum) {

    // Inverted Rastrigin -- multimodal with global max at (2, 2), f* = 0.5.
    // MINGO finds the global optimum for this seed within 10000 evals.
    const auto func = [](const hvector<2>& v) -> double {
        double x = v[0] - 2.0;
        double y = v[1] - 2.0;
        constexpr double DPI = 6.283185307179586476925286766559;

        double rastrigin = -20.0 + (10.0 * cos(DPI * x) - x * x) + (10.0 * cos(DPI * y) - y * y);

        return 1.0 / (1.0 + std::exp(-0.25 * rastrigin));
        };

    hrect rect({
        std::make_pair(-3.0, 3.0),
        std::make_pair(-3.0, 3.0)
        });

    mingo_base<2> algo(func, rect, 23445);
    auto res = algo.result(10000);

    EXPECT_LE(std::abs(res.first[0] - 2.0), 0.1);
    EXPECT_LE(std::abs(res.first[1] - 2.0), 0.1);
    EXPECT_LE(std::abs(res.second - 0.5), 0.01);
}

TEST(MingoBaseTests, RastriginGlobalOptimumMajorityRun) {

    // Robustness check: across several independent seeds, MINGO should
    // land on the global optimum on the majority of runs. This mirrors
    // the evaluation protocol in the paper (30 independent runs, report
    // mean). We use a small number of seeds here (6) and require at
    // least 4/6 to succeed -- a permissive but non-trivial bar.
    const auto func = [](const hvector<2>& v) -> double {
        double x = v[0] - 2.0;
        double y = v[1] - 2.0;
        constexpr double DPI = 6.283185307179586476925286766559;

        double rastrigin = -20.0 + (10.0 * cos(DPI * x) - x * x) + (10.0 * cos(DPI * y) - y * y);

        return 1.0 / (1.0 + std::exp(-0.25 * rastrigin));
        };

    hrect rect({
        std::make_pair(-3.0, 3.0),
        std::make_pair(-3.0, 3.0)
        });

    const uint64_t seeds[] = { 1, 2, 3, 4, 5, 6 };
    int global_hits = 0;
    for (uint64_t s : seeds) {
        mingo_base<2> algo(func, rect, s);
        auto res = algo.result(10000);
        if (res.second >= 0.49) ++global_hits;  // within 0.01 of f* = 0.5
    }

    EXPECT_GE(global_hits, 4);
}