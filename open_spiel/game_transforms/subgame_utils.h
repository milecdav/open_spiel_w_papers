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

#include <array>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
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

// Information about a root state in a subgame.
// Used by both resolving gadget and max-margin gadget.
struct SubgameRoot {
  std::unique_ptr<State> state;
  std::string info_state_string;  // Info state of resolving player
  double reach_prob;              // Reach prob of non-resolving player (π_{-res})
};

// Build SubgameRoots from a set of states.
// For each state, clones it, looks up reach_prob by HistoryString(),
// and skips states with zero reach.
std::vector<SubgameRoot> BuildSubgameRoots(
    const std::vector<std::unique_ptr<State>>& states,
    Player resolving_player,
    const std::unordered_map<std::string, double>& reach_probs);

// A decomposition of a game into trunk + subgames.
// Holds all the data needed to re-solve subgames with any gadget type.
struct SubgameDecomposition {
  std::shared_ptr<const Game> game;
  // Trunk info states per player
  std::unordered_map<int, std::unordered_set<std::string>> trunk_info_states;
  // Subgames grouped by public observation
  std::unordered_map<std::string, std::vector<std::unique_ptr<State>>>
      grouped_subgames;
  // Reach probs per player (index 0 = player 0, index 1 = player 1)
  std::array<std::unordered_map<std::string, double>, 2> reach_probs;
  // Counterfactual values per player
  std::array<std::unordered_map<std::string, double>, 2> cfvs;
};

// Decompose a game at a round boundary (chance-node count).
// Computes reach probs and CFVs for both players using the given policy.
SubgameDecomposition DecomposeGameAtRound(
    std::shared_ptr<const Game> game,
    const Policy& policy,
    int round);

// Decompose a game at a depth boundary (player-action count).
// Computes reach probs and CFVs for both players using the given policy.
SubgameDecomposition DecomposeGameAtDepth(
    std::shared_ptr<const Game> game,
    const Policy& policy,
    int depth);

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

// Collect info state strings at all player nodes where fewer than `round`
// chance nodes have been encountered on the path from root.
// Compare with CollectStatesAtRound which returns State objects *at* a round;
// this returns info state strings for all player nodes *before* it.
// Returns a map from player to the set of their info state strings.
std::unordered_map<int, std::unordered_set<std::string>>
CollectInfoStateStringsBeforeRound(const Game& game, int round);

// Collect info state strings at all player nodes where the action depth
// (number of player actions from root) is strictly less than `depth_limit`.
// Compare with CollectStatesAtDepth which returns State objects *at* a depth;
// this returns info state strings for all player nodes *before* it.
// Chance actions do not count towards the depth.
// Returns a map from player to the set of their info state strings.
std::unordered_map<int, std::unordered_set<std::string>>
CollectInfoStateStringsBeforeDepth(const Game& game, int depth_limit);

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

// Group states by their public observation string.
// Uses kPublicStateObsType (perfect-recall, public only, no private info)
// to obtain observation strings, then groups states with identical observations.
// This is the general-purpose equivalent of game-specific grouping functions
// (e.g., grouping Leduc states by betting sequence).
std::unordered_map<std::string, std::vector<std::unique_ptr<State>>>
GroupStatesByPublicObservation(
    const Game& game,
    std::vector<std::unique_ptr<State>> states);

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

// Collect info state strings for each player by traversing from subgame roots.
// Returns an array of 2 sets (one per player) containing the info state strings
// encountered at that player's decision nodes in the subtrees rooted at `roots`.
std::array<std::unordered_set<std::string>, 2>
CollectSubgameInfoStatesPerPlayer(
    const std::vector<std::unique_ptr<State>>& roots);

}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAME_TRANSFORMS_SUBGAME_UTILS_H_
