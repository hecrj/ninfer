#include "runtime/engine/max_context.h"

#include <iostream>
#include <stdexcept>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

bool rejects(const auto& operation) {
    try {
        operation();
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

} // namespace

int main() {
    int failures = 0;

    // Plain threshold predicate: the boundary is exact.
    {
        const auto feasible = [](std::uint32_t v) { return v <= 42; };
        failures += check(ninfer::runtime::largest_feasible_value(1, 100, feasible) == 42,
                          "threshold predicate did not select the boundary value");
    }

    // The whole range is feasible: the upper bound wins.
    {
        const auto feasible = [](std::uint32_t v) { return v <= 100; };
        failures += check(ninfer::runtime::largest_feasible_value(1, 100, feasible) == 100,
                          "feasible upper bound was not returned");
    }

    // Only the lower bound is feasible.
    {
        const auto feasible = [](std::uint32_t v) { return v <= 1; };
        failures += check(ninfer::runtime::largest_feasible_value(1, 100, feasible) == 1,
                          "feasible lower bound was not returned");
    }

    // Single-value range.
    {
        const auto feasible = [](std::uint32_t) { return true; };
        failures += check(ninfer::runtime::largest_feasible_value(7, 7, feasible) == 7,
                          "single-value range was not returned");
    }

    // Page-quantized step predicate on a model-scale range: the KV floor is ceil(v/64) pages, so
    // ten feasible pages bound the context at 640 tokens.
    {
        const auto feasible = [](std::uint32_t v) { return (v + 63) / 64 <= 10; };
        failures += check(ninfer::runtime::largest_feasible_value(1, 262144, feasible) == 640,
                          "page-quantized predicate did not select the last feasible page");
    }

    // The minimum feasible value itself is a valid answer (low == high boundary).
    {
        const auto feasible = [](std::uint32_t v) { return v <= 2048; };
        failures += check(ninfer::runtime::largest_feasible_value(2048, 262144, feasible) == 2048,
                          "answer at the search lower bound was missed");
    }

    // Inverted bounds are rejected.
    failures += check(rejects([] {
                          const auto feasible = [](std::uint32_t) { return true; };
                          (void)ninfer::runtime::largest_feasible_value(10, 1, feasible);
                      }),
                      "inverted feasibility bounds were accepted");

    // No feasible value in the range is rejected.
    failures += check(rejects([] {
                          const auto feasible = [](std::uint32_t) { return false; };
                          (void)ninfer::runtime::largest_feasible_value(1, 100, feasible);
                      }),
                      "an all-infeasible range was accepted");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
