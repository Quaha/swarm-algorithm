#pragma once

#include "hvector.hpp"
#include <vector>
#include <iostream>
#include <fstream>

namespace swarm_algorithm {

    template<size_t DIM>
    void printBestsToCSV(const std::vector<std::tuple<size_t, hvector<DIM>, double>>& best_updates) {
        // Заголовок
        std::cout << "Iteration";
        for (size_t i = 0; i < DIM; i++) {
            std::cout << ",X" << i;
        }
        std::cout << ",Fitness\n";

        // Данные
        for (const auto& record : best_updates) {
            size_t iteration = std::get<0>(record);
            const auto& position = std::get<1>(record);
            double fitness = std::get<2>(record);

            std::cout << iteration;
            for (size_t i = 0; i < DIM; i++) {
                std::cout << "," << std::setprecision(10) << position[i];
            }
            std::cout << "," << std::setprecision(10) << fitness << "\n";
        }
    }

    template<size_t DIM>
    void saveBestsToCSV(const std::vector<std::tuple<size_t, hvector<DIM>, double>>& best_updates,
        const std::string& filename) {
        std::ofstream file(filename);

        if (!file.is_open()) {
            std::cerr << "Error: Cannot open file " << filename << std::endl;
            return;
        }

        // Заголовок
        file << "Iteration";
        for (size_t i = 0; i < DIM; i++) {
            file << ",X" << i;
        }
        file << ",Fitness\n";

        // Данные
        for (const auto& record : best_updates) {
            size_t iteration = std::get<0>(record);
            const auto& position = std::get<1>(record);
            double fitness = std::get<2>(record);

            file << iteration;
            for (size_t i = 0; i < DIM; i++) {
                file << "," << std::setprecision(10) << position[i];
            }
            file << "," << std::setprecision(10) << fitness << "\n";
        }

        file.close();
        std::cout << "Data saved to " << filename << std::endl;
    }
}