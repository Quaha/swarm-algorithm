#include <gtest/gtest.h>
#include <random>
#include <cmath>
#include <vector>
#include <algorithm>

#include "hrect.hpp"
#include "truncated_normal.hpp"

using namespace swarm_algorithm;

class FastTruncatedNormalAdapterTest : public ::testing::Test {
protected:
    void SetUp() override {
        gen_.seed(42);
    }

    std::mt19937 gen_;

protected:
    double normal_pdf(double x, double mean, double stddev) {
        constexpr double DPI = 6.283185307179586476925286766559;
        return std::exp(-0.5 * (x - mean) * (x - mean) / (stddev * stddev)) / (std::sqrt(DPI) * stddev);
    }
};

TEST_F(FastTruncatedNormalAdapterTest, ValuesWithinBounds) {
    double mean = 0.0;
    double stddev = 1.0;
    double lower = -2.0;
    double upper = 2.0;

    hrect search_area({
        std::make_pair(lower, upper)
        });

    truncated_normal adapter(search_area, gen_, 1);

    const int N = 100000;
    for (int i = 0; i < N; ++i) {
        double value = adapter.generate(mean, stddev, 0);
        EXPECT_GE(value, lower - 1e-9);
        EXPECT_LE(value, upper + 1e-9);
    }
}

TEST_F(FastTruncatedNormalAdapterTest, SampleMean) {
    double mean = 5.0;
    double stddev = 2.0;
    double lower = 0.0;
    double upper = 10.0;

    hrect search_area({
        std::make_pair(lower, upper)
        });

    truncated_normal adapter(search_area, gen_, 1);

    const int N = 100000;
    double sum = 0.0;

    for (int i = 0; i < N; ++i) {
        sum += adapter.generate(mean, stddev, 0);
    }

    double sample_mean = sum / N;
    double expected_mean = mean;

    EXPECT_NEAR(sample_mean, expected_mean, 0.05);
}

TEST_F(FastTruncatedNormalAdapterTest, SampleVariance) {
    double mean = 0.0;
    double stddev = 1.0;
    double lower = -3.0;
    double upper = 3.0;

    hrect search_area({
        std::make_pair(lower, upper)
        });

    truncated_normal adapter(search_area, gen_, 1);

    const int N = 100000;
    double sum = 0.0;
    double sum_sq = 0.0;

    for (int i = 0; i < N; ++i) {
        double value = adapter.generate(mean, stddev, 0);
        sum += value;
        sum_sq += value * value;
    }

    double sample_mean = sum / N;
    double sample_var = (sum_sq / N) - (sample_mean * sample_mean);

    EXPECT_NEAR(sample_var, 0.97, 0.02);
}

TEST_F(FastTruncatedNormalAdapterTest, VeryNarrowInterval) {
    double mean = 0.0;
    double stddev = 1.0;
    double lower = -0.1;
    double upper = 0.1;

    hrect search_area({
        std::make_pair(lower, upper)
        });

    truncated_normal adapter(search_area, gen_, 1);

    const int N = 10000;
    double sum = 0.0;

    for (int i = 0; i < N; ++i) {
        double value = adapter.generate(mean, stddev, 0);
        EXPECT_GE(value, lower - 1e-9);
        EXPECT_LE(value, upper + 1e-9);
        sum += value;
    }

    double sample_mean = sum / N;
    EXPECT_NEAR(sample_mean, (lower + upper) / 2, 0.01);
}

TEST_F(FastTruncatedNormalAdapterTest, VerySmallStddev) {
    double mean = 0.0;
    double stddev = 0.01;
    double lower = -1.0;
    double upper = 1.0;

    hrect search_area({
        std::make_pair(lower, upper)
        });

    truncated_normal adapter(search_area, gen_, 1);

    const int N = 10000;
    double sum = 0.0;

    for (int i = 0; i < N; ++i) {
        double value = adapter.generate(mean, stddev, 0);
        EXPECT_NEAR(value, mean, 0.05);
        sum += value;
    }

    double sample_mean = sum / N;
    EXPECT_NEAR(sample_mean, mean, 0.001);
}

TEST_F(FastTruncatedNormalAdapterTest, VeryLargeStddev) {
    double mean = 0.0;
    double stddev = 100.0;
    double lower = -1.0;
    double upper = 1.0;

    hrect search_area({
        std::make_pair(lower, upper)
        });

    truncated_normal adapter(search_area, gen_, 1);

    const int N = 10000;
    for (int i = 0; i < N; ++i) {
        double value = adapter.generate(mean, stddev, 0);
        EXPECT_GE(value, lower - 1e-9);
        EXPECT_LE(value, upper + 1e-9);
    }
}

TEST_F(FastTruncatedNormalAdapterTest, BoundaryValues) {
    double mean = 0.0;
    double stddev = 1.0;
    double lower = -1.0;
    double upper = 1.0;

    hrect search_area({
        std::make_pair(lower, upper)
        });

    truncated_normal adapter(search_area, gen_, 1);

    const int N = 100000;
    int near_lower = 0;
    int near_upper = 0;
    const double epsilon = 1e-6;

    for (int i = 0; i < N; ++i) {
        double value = adapter.generate(mean, stddev, 0);

        if (std::abs(value - lower) < epsilon) near_lower++;
        if (std::abs(value - upper) < epsilon) near_upper++;

        EXPECT_GE(value, lower - epsilon);
        EXPECT_LE(value, upper + epsilon);
    }

    EXPECT_LT(near_lower, N / 1000);
    EXPECT_LT(near_upper, N / 1000);
}