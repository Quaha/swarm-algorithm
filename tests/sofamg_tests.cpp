#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include "hvector.hpp"
#include "sofamg.hpp"

using namespace swarm_algorithm;

TEST(SofamgTests, SphereConvergence) {
    const auto func = [](const hvector<2>& v) -> double {
        double x = v[0] - 1.0;
        double y = v[1] - 1.0;
        return 1.0 / (1.0 + x * x + y * y);
        };

    hrect rect({
        std::make_pair(-3.0, 3.0),
        std::make_pair(-3.0, 3.0)
        });

    sofamg_base<2> algo(func, rect, 23445);
    auto res = algo.result(10000);

    EXPECT_LE(std::abs(res.first[0] - 1.0), 0.02);
    EXPECT_LE(std::abs(res.first[1] - 1.0), 0.02);
    EXPECT_GE(res.second, 0.999);
}

TEST(SofamgTests, RastriginFindsGlobalOptimum) {
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

    sofamg_base<2> algo(func, rect, 23445);
    auto res = algo.result(10000);

    EXPECT_LE(std::abs(res.first[0] - 2.0), 0.05);
    EXPECT_LE(std::abs(res.first[1] - 2.0), 0.05);
    EXPECT_LE(std::abs(res.second - 0.5), 0.01);
}

TEST(SofamgTests, RastriginRobustness) {
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
        sofamg_base<2> algo(func, rect, s);
        auto res = algo.result(10000);
        if (res.second >= 0.499) ++global_hits;
    }

    EXPECT_GE(global_hits, 5);
}

TEST(SofamgTests, CustomSchedules) {
    const auto func = [](const hvector<2>& v) -> double {
        double x = v[0] - 1.0;
        double y = v[1] - 1.0;
        return 1.0 / (1.0 + x * x + y * y);
        };

    hrect rect({
        std::make_pair(-3.0, 3.0),
        std::make_pair(-3.0, 3.0)
        });

    sofamg_base<2> algo(func, rect, 23445);
    algo.set_varsigma([](size_t k) -> double {
        if (k < 1) k = 1;
        return std::pow(1.0 / static_cast<double>(k), 0.35);
        });
    algo.set_phi([](size_t k) -> double {
        return std::sqrt(static_cast<double>(k) * 0.005 + 1.0);
        });

    auto res = algo.result(10000);
    EXPECT_LE(std::abs(res.first[0] - 1.0), 0.05);
    EXPECT_LE(std::abs(res.first[1] - 1.0), 0.05);
    EXPECT_GE(res.second, 0.99);
}
