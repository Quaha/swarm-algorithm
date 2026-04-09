#include <iostream>

#include "module.hpp"

extern "C" {
#include "cec17.h"
}

int main() {

    int a, b;
    std::cin >> a >> b;

    std::cout << add(a, b);

    return 0;
}