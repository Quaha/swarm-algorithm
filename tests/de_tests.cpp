#include <gtest/gtest.h>
#include <cmath>
#include "de.hpp"
#include "hvector.hpp"

using namespace swarm_algorithm;

// DE/current-to-best/1/bin balances exploitation (pull toward x_best)
// with exploration (difference vector F*(x_r1 - x_r2)). Unlike CRS, it
// reliably finds the global optimum of moderately multimodal functions
// in low dimensions. The tests below verify both behaviours:
//   1. SphereConvergence                -- unimodal
//   2. RastriginFindsGlobalOptimum      -- multimodal (2D, global reachable)
//   3. CrGreedyStillConverges           -- sanity check for CR = 1.0

TEST(DeBaseTests, SphereConvergence) {

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

    de_base<2> algo(func, rect, 23445);
    auto res = algo.result(10000);

    EXPECT_LE(std::abs(res.first[0] - 1.0), 0.01);
    EXPECT_LE(std::abs(res.first[1] - 1.0), 0.01);
    EXPECT_GE(res.second, 0.9999);
}

TEST(DeBaseTests, RastriginFindsGlobalOptimum) {

    // Inverted Rastrigin -- multimodal with global max at (2, 2), f* = 0.5.
    // DE/current-to-best locates (2, 2) within 10000 evals for the 2D case.
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

    de_base<2> algo(func, rect, 23445);
    auto res = algo.result(10000);

    EXPECT_LE(std::abs(res.first[0] - 2.0), 0.1);
    EXPECT_LE(std::abs(res.first[1] - 2.0), 0.1);
    EXPECT_LE(std::abs(res.second - 0.5), 0.01);
}

TEST(DeBaseTests, CrGreedyStillConverges) {

    // CR = 1.0 turns binomial crossover into "copy all coordinates from
    // the mutant". The j_rand trick then becomes a no-op. This is a
    // corner case in the crossover code path; verify it still converges
    // on a simple unimodal problem.
    const auto func = [](const hvector<2>& v) -> double {
        double x = v[0] - 1.0;
        double y = v[1] - 1.0;
        return 1.0 / (1.0 + x * x + y * y);
        };

    hrect rect({
        std::make_pair(-3.0, 3.0),
        std::make_pair(-3.0, 3.0)
        });

    de_base<2> algo(func, rect, 23445);
    algo.set_CR(1.0);
    auto res = algo.result(10000);

    EXPECT_LE(std::abs(res.first[0] - 1.0), 0.01);
    EXPECT_LE(std::abs(res.first[1] - 1.0), 0.01);
    EXPECT_GE(res.second, 0.9999);
}