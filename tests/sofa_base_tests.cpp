#include <gtest/gtest.h>
#include <cmath>
#include "sofa_base.hpp"

using namespace swarm_algorithm;

TEST(SofaBaseTests, SofaBase) {

    // Rastrigin inverted
    const auto func = [](const std::vector<double> &v) -> double {
        double x = v[0];
        double y = v[1];
        constexpr double DPI = 6.283185307179586476925286766559;
        return std::max(0.0, - (x * x - 10 * cos(DPI * x)) - (y * y - 10 * cos(DPI * y)));
        };

    hrect rect({
        std::make_pair(-3.0, 3.0),
        std::make_pair(-3.0, 3.0)
        });

    sofa_base algo(func, rect);

    auto res = algo.result(10000);
    
    EXPECT_LE(abs(res.first[0] - 0.0), 0.1);
    EXPECT_LE(abs(res.first[1] - 0.0), 0.1);
    EXPECT_LE(abs(res.second - 20.0), 0.1);
}