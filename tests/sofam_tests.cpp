#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include "hvector.hpp"
#include "sofam.hpp"

using namespace swarm_algorithm;

// SoFAM improves on SoFA by replacing the base SoFA's isotropic normal
// mutations with an anisotropic truncated-Cauchy mutation operator, with
// a shifted-centre "strong" branch governed by Q and H. In 2D it is
// noticeably more accurate than base SoFA and competitive with DE/MINGO.
// Tests:
//   1. SphereConvergence                  -- unimodal, strict
//   2. RastriginFindsGlobalOptimum        -- multimodal, known-good seed
//   3. RastriginRobustness                -- multi-seed statistical check
//   4. CustomSchedules                    -- control functions can be
//                                            overridden without breaking
//                                            convergence on unimodal f

TEST(SofamTests, SphereConvergence) {

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

    sofam_base<2> algo(func, rect, 23445);
    auto res = algo.result(10000);

    // SoFAM is stochastic -- slightly looser tolerances than CRS/DE.
    EXPECT_LE(std::abs(res.first[0] - 1.0), 0.02);
    EXPECT_LE(std::abs(res.first[1] - 1.0), 0.02);
    EXPECT_GE(res.second, 0.999);
}

TEST(SofamTests, RastriginFindsGlobalOptimum) {

    // Inverted Rastrigin -- multimodal, global max at (2, 2), f* = 0.5.
    // Unlike base SoFA (which typically locks onto the basin around
    // (2, 2) but cannot get below the 0.01 tolerance), SoFAM's
    // anisotropic mutations reach the global optimum precisely.
    const auto func = [](const hvector<2>& v) -> double {
        double x = v[0] - 2.0;
        double y = v[1] - 2.0;
        constexpr double DPI = 6.283185307179586476925286766559;

        double rastrigin = -20.0 + (10.0 * cos(DPI * x) - x * x)
                                 + (10.0 * cos(DPI * y) - y * y);

        return 1.0 / (1.0 + std::exp(-0.25 * rastrigin));
        };

    hrect rect({
        std::make_pair(-3.0, 3.0),
        std::make_pair(-3.0, 3.0)
        });

    sofam_base<2> algo(func, rect, 23445);
    auto res = algo.result(10000);

    EXPECT_LE(std::abs(res.first[0] - 2.0), 0.05);
    EXPECT_LE(std::abs(res.first[1] - 2.0), 0.05);
    EXPECT_LE(std::abs(res.second - 0.5), 0.01);
}

TEST(SofamTests, RastriginRobustness) {

    // Across multiple seeds, SoFAM should find the global optimum with
    // high reliability -- this mirrors how the paper evaluates the
    // algorithm (multiple independent runs, report success rate). The
    // algorithm reached f >= 0.499 on all six seeds in our experiments,
    // so we require at least 5/6 here as a permissive but meaningful bar.
    const auto func = [](const hvector<2>& v) -> double {
        double x = v[0] - 2.0;
        double y = v[1] - 2.0;
        constexpr double DPI = 6.283185307179586476925286766559;

        double rastrigin = -20.0 + (10.0 * cos(DPI * x) - x * x)
                                 + (10.0 * cos(DPI * y) - y * y);

        return 1.0 / (1.0 + std::exp(-0.25 * rastrigin));
        };

    hrect rect({
        std::make_pair(-3.0, 3.0),
        std::make_pair(-3.0, 3.0)
        });

    const uint64_t seeds[] = { 23445, 42, 1337, 7, 99, 2024 };
    int global_hits = 0;
    for (uint64_t s : seeds) {
        sofam_base<2> algo(func, rect, s);
        auto res = algo.result(10000);
        if (res.second >= 0.499) ++global_hits;
    }

    EXPECT_GE(global_hits, 5);
}

TEST(SofamTests, CustomSchedules) {

    // Sanity check: overriding varsigma(k) and phi(k) should not break
    // convergence on a simple unimodal problem.
    const auto func = [](const hvector<2>& v) -> double {
        double x = v[0] - 1.0;
        double y = v[1] - 1.0;
        return 1.0 / (1.0 + x * x + y * y);
        };

    hrect rect({
        std::make_pair(-3.0, 3.0),
        std::make_pair(-3.0, 3.0)
        });

    sofam_base<2> algo(func, rect, 23445);
    // Slightly faster-decaying varsigma than the theorem's rate.
    algo.set_varsigma([](size_t k) -> double {
        if (k < 1) k = 1;
        return std::pow(1.0 / static_cast<double>(k), 0.35);
        });
    // More aggressive selection pressure.
    algo.set_phi([](size_t k) -> double {
        return std::sqrt(static_cast<double>(k) * 0.005 + 1.0);
        });

    auto res = algo.result(10000);
    EXPECT_LE(std::abs(res.first[0] - 1.0), 0.05);
    EXPECT_LE(std::abs(res.first[1] - 1.0), 0.05);
    EXPECT_GE(res.second, 0.99);
}
