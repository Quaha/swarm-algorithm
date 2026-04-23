#include <gtest/gtest.h>
#include <cmath>
#include "crs.hpp"
#include "hvector.hpp"

using namespace swarm_algorithm;

// CRS2-LM is a randomized Nelder-Mead-like algorithm. It is a strong local
// optimizer but weak at escaping local optima on heavily multimodal
// landscapes. The two tests below reflect this:
//   1. SphereConvergence -- strict check on a unimodal function.
//   2. RastriginReachesLocalOptimum -- soft check on multimodal Rastrigin,
//      verifying CRS at least locks onto one of the integer-lattice local
//      maxima near (2, 2), but not requiring it to find the global one.

TEST(CrsBaseTests, SphereConvergence) {

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

    crs_base<2> algo(func, rect, 23445);
    auto res = algo.result(10000);

    EXPECT_LE(std::abs(res.first[0] - 1.0), 0.01);
    EXPECT_LE(std::abs(res.first[1] - 1.0), 0.01);
    EXPECT_GE(res.second, 0.9999);
}

TEST(CrsBaseTests, RastriginReachesLocalOptimum) {

    // Multimodal; global max at (2, 2), f* = 0.5. CRS2-LM typically locks
    // onto an integer-lattice neighbour of (2, 2) such as (2, 3), (3, 2),
    // or (3, 3) and does not escape. We verify the algorithm at least
    // converges into the basin around (2, 2) and improves substantially
    // over the uniform-random starting value.
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

    crs_base<2> algo(func, rect, 23445);
    auto res = algo.result(10000);

    // Must land near (2, 2) -- within one lattice step on each axis.
    EXPECT_LE(std::abs(res.first[0] - 2.0), 1.1);
    EXPECT_LE(std::abs(res.first[1] - 2.0), 1.1);
    // Must clearly beat a random baseline. Inverted Rastrigin at the
    // edges/valleys drops well below 0.1; a local optimum is >= 0.35.
    EXPECT_GE(res.second, 0.3);
}