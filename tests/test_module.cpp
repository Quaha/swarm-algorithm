#include <gtest/gtest.h>

#include "module.hpp"

TEST(Add, CanUseFunction) {
    EXPECT_NO_THROW(add(1, 1));
}

TEST(Add, IsResultCorrect) {
    EXPECT_EQ(add(1, 2), 3);
}
