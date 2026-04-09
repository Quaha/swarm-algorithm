#include <gtest/gtest.h>
#include <cmath>
#include "sofa_base.hpp"
#include "hvector.hpp"

extern "C" {
#include "cec17.h"
}

using namespace swarm_algorithm;

TEST(SofaBaseTests, SofaBase) {

    // Rastrigin inverted
    const auto func = [](const hvector<2> &v) -> double {
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
    
    sofa_base<2> algo(func, rect, 23445);
    algo.reserve_buffers(12000);
    auto res = algo.result(10000);
    
    EXPECT_LE(abs(res.first[0] - 2.0), 0.1);
    EXPECT_LE(abs(res.first[1] - 2.0), 0.1);
    EXPECT_LE(abs(res.second - 0.5), 0.01);
}