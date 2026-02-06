// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OPEN_SPIEL_GAME_TRANSFORMS_SUBGAME_UTILS_H_
#define OPEN_SPIEL_GAME_TRANSFORMS_SUBGAME_UTILS_H_

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"

// Utility functions for subgame decomposition:
//   - Collecting states at specific depths or rounds
//   - Computing counterfactual values at states
//   - Computing reach probabilities
//
// These are general-purpose helpers with no dependency on resolving_gadget
// or unsafe_subgame. They can be used by any code that needs to decompose
// a game into subgames.

namespace open_spiel {

// Collect non-terminal states where `predicate` returns true.
// Traverses the game tree from the initial state. When predicate(state)
// is true, the state is collected and recursion stops (the state forms
// a "frontier"). Terminal states are never collected.
std::vector<std::unique_ptr<State>> CollectStates(
    const Game& game,
    const std::function<bool(const State&)>& predicate);

// Collect all states at a specific depth measured in player actions
// (chance actions do not count). For example, depth=2 collects all
// non-terminal states reached after exactly 2 player actions.
std::vector<std::unique_ptr<State>> CollectStatesAtDepth(
    const Game& game, int depth);

// Collect the first non-chance states after exactly `round` chance nodes
// have been encountered along each path. This corresponds to the start of
// a new "round" in games where rounds are delimited by chance events
// (e.g., card deals in poker).
//
// Example: In Leduc poker, round=3 collects the round-2 root states
// (after 2 private cards + 1 public card have been dealt).
std::vector<std::unique_ptr<State>> CollectStatesAtRound(
    const Game& game, int round);

// Compute counterfactual values at specific states with explicit reach probs.
// CFV(I) = sum_{h in I} pi_{-i}(h) * v(h)
// where pi_{-i}(h) is the opponent's reach probability and v(h) is the
// expected value at state h under the given policy.
//
// Parameters:
//   game: The game (used for structure only)
//   policy: Joint policy for computing expected values
//   player: Player whose CFVs we compute
//   states: States at which to compute CFVs
//   reach_probs: Map from state history string to opponent reach probability
//
// Returns: Map from info state string to counterfactual value.
std::unordered_map<std::string, double> ComputeCounterfactualValuesAtStates(
    const Game& game,
    const Policy& policy,
    Player player,
    const std::vector<const State*>& states,
    const std::unordered_map<std::string, double>& reach_probs);

// Convenience overload: compute expected value at one state per info set.
// WARNING: This does NOT compute proper counterfactual values since it
// ignores reach probabilities and multiple states per info set.
// Only use for simple testing, not for proper gadget construction.
std::unordered_map<std::string, double> ComputeCounterfactualValuesAtStates(
    const Game& game,
    const Policy& opponent_policy,
    Player player,
    const std::vector<const State*>& states);

// Compute reach probabilities for a player at a set of target states.
// The reach probability includes both the player's own action probabilities
// and chance probabilities along the path.
//
// Parameters:
//   game: The game
//   opponent_policy: Policy used to compute action probabilities
//   opponent: The player whose reach probabilities we compute
//   states: Target states at which to compute reach probs
//
// Returns: Map from state history string to reach probability.
std::unordered_map<std::string, double> ComputeReachProbabilities(
    const Game& game,
    const Policy& opponent_policy,
    Player opponent,
    const std::vector<const State*>& states);

}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAME_TRANSFORMS_SUBGAME_UTILS_H_
