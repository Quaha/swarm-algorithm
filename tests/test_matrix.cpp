#include <gtest/gtest.h>

#include "matrix.hpp"
#include "utils.hpp"

TEST(Matrix, CanCreateCorrectMatrix) {
    EXPECT_NO_THROW(Matrix(5));
}

TEST(Matrix, CantCreateIncorrectMatrix) {
    EXPECT_ANY_THROW(Matrix(-4));
}

TEST(Matrix, CanAccessElement) {
    Matrix m(3);

    EXPECT_NO_THROW(m(1, 1));
}

TEST(Matrix, CantAccessElementWithIncorrectIndex) {
    Matrix m(3);

    EXPECT_ANY_THROW(m(3, 1));
    EXPECT_ANY_THROW(m(1, 3));
    EXPECT_ANY_THROW(m(-1, 0));
    EXPECT_ANY_THROW(m(0, -1));
}

TEST(Matrix, CanModifyElement) {
    Matrix m(2);

    EXPECT_NO_THROW(m(0, 0) = 5.0);
    EXPECT_NEAR(m(0, 0), 5.0, EPS);
}

TEST(Matrix, CantModifyElementWithIncorrectIndex) {
    Matrix m(2);

    EXPECT_ANY_THROW(m(2, 0) = 3.0);
    EXPECT_ANY_THROW(m(0, 2) = 3.0);
    EXPECT_ANY_THROW(m(-1, 0) = 3.0);
    EXPECT_ANY_THROW(m(0, -1) = 3.0);
}

TEST(Matrix, CanGetSize) {
    Matrix m(3);

    EXPECT_EQ(m.size(), 3);
}

TEST(Matrix, FillMatrixAndSaveParameters) {
    Matrix m(5);

    m.fillMatrix(
        0.12, 0.22,     // alpha_min, alpha_max
        0.91, 0.95,     // beta_1, beta_2
        true,           // concentrated
        false,          // maturation
        true,           // inorganic
        3,              // v
        1.1             // beta_max
    );

    EXPECT_NEAR(m.getAlphaMin(), 0.12, EPS);
    EXPECT_NEAR(m.getAlphaMax(), 0.22, EPS);
    EXPECT_NEAR(m.getBeta1(), 0.91, EPS);
    EXPECT_NEAR(m.getBeta2(), 0.95, EPS);
    EXPECT_TRUE(m.isConcentrated());
    EXPECT_FALSE(m.hasMaturation());
    EXPECT_TRUE(m.isInorganic());
    EXPECT_EQ(m.getV(), 3);
    EXPECT_NEAR(m.getBetaMax(), 1.1, EPS);
}

TEST(Matrix, FillMatrixAlphaColumnInRange) {
    Matrix m(10);

    double alpha_min = 0.12;
    double alpha_max = 0.22;

    for (int test_case = 0; test_case < 10'000; ++test_case) {
        m.fillMatrix(alpha_min, alpha_max, 0.91, 0.95, false, false, false, 0, 1.1);

        for (int i = 0; i < 10; ++i) {
            double x = m(i, 0);
            EXPECT_GE(x, alpha_min);
            EXPECT_LE(x, alpha_max);
        }
    }
}

TEST(Matrix, FillMatrixRowsDecreaseInRangeNoConcentrationNoMaturation) {
    Matrix m(10);

    double alpha_min = 0.12;
    double alpha_max = 0.22;

    double beta_1 = 0.91;
    double beta_2 = 0.95;

    int v = 0;

    for (int test_case = 0; test_case < 10'000; ++test_case) {
        m.fillMatrix(alpha_min, alpha_max, beta_1, beta_2, false, false, false, v, 1.1);

        for (int row = 0; row < 10; ++row) {
            for (int col = 1; col < 10; ++col) {
                double prev_value = m(row, col - 1);
                double curr_value = m(row, col);

                EXPECT_GE(curr_value, prev_value * beta_1);
                EXPECT_LE(curr_value, prev_value * beta_2);
            }
        }
    }
}

TEST(Matrix, FillMatrixRowsDecreaseInRangeWithConcentrationNoMaturation) {
    Matrix m(10);

    double alpha_min = 0.12;
    double alpha_max = 0.22;

    double beta_1 = 0.91;
    double beta_2 = 0.95;

    int v = 0;

    for (int test_case = 0; test_case < 10'000; ++test_case) {
        m.fillMatrix(alpha_min, alpha_max, beta_1, beta_2, true, false, false, v, 1.1);

        for (int row = 0; row < 10; ++row) {
            for (int col = 1; col < 10; ++col) {
                double prev_value = m(row, col - 1);
                double curr_value = m(row, col);

                EXPECT_GE(curr_value, prev_value * beta_1);
                EXPECT_LE(curr_value, prev_value * beta_2);
            }
        }
    }
}

TEST(Matrix, FillMatrixRowsDecreaseInRangeWithConcentrationWithMaturation) {
    Matrix m(10);

    double alpha_min = 0.12;
    double alpha_max = 0.22;

    double beta_1 = 0.91;
    double beta_2 = 0.95;

    double beta_max = 1.1;

    int v = 5;

    for (int test_case = 0; test_case < 10'000; ++test_case) {
        m.fillMatrix(alpha_min, alpha_max, beta_1, beta_2, true, true, false, v, beta_max);

        for (int row = 0; row < 10; ++row) {

            for (int col = 1; col <= v; ++col) {
                double prev_value = m(row, col - 1);
                double curr_value = m(row, col);

                EXPECT_GE(curr_value, prev_value);
				EXPECT_LE(curr_value, prev_value * beta_max);
            }

            for (int col = v + 1; col < 10; ++col) {
                double prev_value = m(row, col - 1);
                double curr_value = m(row, col);

                EXPECT_GE(curr_value, prev_value * beta_1);
                EXPECT_LE(curr_value, prev_value * beta_2);
            }
        }
    }
}

TEST(Matrix, FillMatrixRowsDecreaseInRangeNoConcentrationWithMaturation) {
    Matrix m(10);

    double alpha_min = 0.12;
    double alpha_max = 0.22;

    double beta_1 = 0.91;
    double beta_2 = 0.95;

    double beta_max = 1.1;

    int v = 5;

    for (int test_case = 0; test_case < 10'000; ++test_case) {
        m.fillMatrix(alpha_min, alpha_max, beta_1, beta_2, false, true, false, v, beta_max);

        for (int row = 0; row < 10; ++row) {

            for (int col = 1; col <= v; ++col) {
                double prev_value = m(row, col - 1);
                double curr_value = m(row, col);

                EXPECT_GE(curr_value, prev_value);
                EXPECT_LE(curr_value, prev_value * beta_max);
            }

            for (int col = v + 1; col < 10; ++col) {
                double prev_value = m(row, col - 1);
                double curr_value = m(row, col);

                EXPECT_GE(curr_value, prev_value * beta_1);
                EXPECT_LE(curr_value, prev_value * beta_2);
            }
        }
    }
}

TEST(Matrix, FillMatrixInorganicEffectReducesValuesButNotBelowZero) {
    Matrix m(10);

    double alpha_min = 0.8;
    double alpha_max = 0.9;

    double beta_1 = 0.91;
    double beta_2 = 0.95;

    double beta_max = 1.1;
    int v = 3;

    for (int test_case = 0; test_case < 10'000; ++test_case) {
        m.fillMatrix(alpha_min, alpha_max, beta_1, beta_2, true, true, true, v, beta_max);

        for (int row = 0; row < 10; ++row) {
            for (int col = 0; col < 10; ++col) {
                EXPECT_GE(m(row, col), 0.0);
                EXPECT_LE(m(row, col), 1.0);
            }
        }
    }
}

TEST(Matrix, CanGetAlphaMin) {
	Matrix m(2);

    EXPECT_NEAR(0.12, m.getAlphaMin(), EPS);
}

TEST(Matrix, CanGetAlphaMax) {
    Matrix m(2);

    EXPECT_NEAR(0.22, m.getAlphaMax(), EPS);
}

TEST(Matrix, CanGetBeta1) {
    Matrix m(2);

    EXPECT_NEAR(0.85, m.getBeta1(), EPS);
}

TEST(Matrix, CanGetBeta2) {
    Matrix m(2);

    EXPECT_NEAR(1.00, m.getBeta2(), EPS);
}

TEST(Matrix, CanGetConcentratedStatus) {
    Matrix m(2);

    EXPECT_FALSE(m.isConcentrated());
}

TEST(Matrix, CanGetMaturationStatus) {
    Matrix m(2);

    EXPECT_FALSE(m.hasMaturation());
}

TEST(Matrix, CanGetInorganicStatus) {
    Matrix m(2);

    EXPECT_FALSE(m.isInorganic());
}

TEST(Matrix, CanGetV) {
    Matrix m(2);

    EXPECT_EQ(0, m.getV());
}

TEST(Matrix, CanGetBetaMax) {
    Matrix m(2);

    EXPECT_NEAR(1.15, m.getBetaMax(), EPS);
}

TEST(Matrix, CanGetStrategyScore) {
    Matrix m(2);
    std::vector<int> seq = { 0, 1 };

    EXPECT_NO_THROW(m.getStrategyScore(seq));
}

TEST(Matrix, CantGetStrategyScoreWithIncorrectSeqNo1) {
    Matrix m(2);
    std::vector<int> seq = { 0, 1, 2 };

    EXPECT_ANY_THROW(m.getStrategyScore(seq));
}

TEST(Matrix, CantGetStrategyScoreWithIncorrectSeqNo2) {
    Matrix m(2);
    std::vector<int> seq = { 3, -1};

    EXPECT_ANY_THROW(m.getStrategyScore(seq));
}

TEST(Matrix, CanGetCorrectStrategyScore) {
    Matrix m(2);
    m(0, 0) = 1.0; m(0, 1) = 3.0;
    m(1, 0) = 2.0; m(1, 1) = 1.5;

    std::vector<int> seq = { 1, 0 };

    EXPECT_NEAR(m.getStrategyScore(seq), 5.0, EPS);
}

