#pragma once

// Monotone search over a startup feasibility predicate. The automatic --max-context resolver
// finds the largest context whose complete runtime layout fits the remaining device memory; the
// predicate is monotone (a larger context never fits when a smaller one does not), so the answer
// is a single boundary that a binary search locates without probing the whole range.

#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninfer::runtime {

// Returns the largest value in [low, high] for which feasible(value) is true.
//
// Preconditions:
//   * low <= high;
//   * feasible is deterministic and monotone: feasible(v) implies feasible(w) for every
//     w <= v (a larger context never fits when a smaller one does not).
//   * feasible(low) is true (a feasible value exists in the range).
//
// The binary search relies on the monotonicity precondition; the caller owns it. Evaluating the
// predicate is expected to be comparatively cheap: the search performs O(log2(high - low))
// evaluations plus at most two boundary evaluations.
template <class Feasible>
[[nodiscard]] std::uint32_t largest_feasible_value(std::uint32_t low, std::uint32_t high,
                                                   Feasible&& feasible) {
    if (low > high) { throw std::invalid_argument("feasibility search bounds are inverted"); }
    if (!feasible(low)) {
        throw std::invalid_argument("no value in the feasibility search range is feasible");
    }
    if (feasible(high)) { return high; }
    std::uint32_t lo = low;  // feasible
    std::uint32_t hi = high; // infeasible
    while (hi - lo > 1) {
        const std::uint32_t mid = lo + (hi - lo) / 2;
        if (feasible(mid)) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

} // namespace ninfer::runtime
