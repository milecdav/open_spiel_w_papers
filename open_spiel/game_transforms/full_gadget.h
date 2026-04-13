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

#ifndef OPEN_SPIEL_GAME_TRANSFORMS_FULL_GADGET_H_
#define OPEN_SPIEL_GAME_TRANSFORMS_FULL_GADGET_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "open_spiel/game_transforms/game_wrapper.h"
#include "open_spiel/game_transforms/subgame_utils.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"

// Full Gadget Game Transform
//
// Instead of using a small approximating gadget (resolving/max-margin),
// the Full Gadget keeps the actual trunk game structure, giving exact
// exploitability measurement.
//
// Game structure:
//   - Trunk phase: The resolving player's decisions become chance nodes
//     (playing according to blueprint). Non-resolving player is free.
//   - At subgame boundary:
//     - Target subgame group -> enter subgame phase (both players free)
//     - Non-target boundary states -> MVS (matrix-valued states) where both
//       players select from strategy portfolios, receiving matrix payoffs.
//       Falls back to scalar expected values if no portfolios are provided.
//
// Two modes:
//   kPath: Only keep trunk states that can reach the target subgame.
//          Other trunk states -> terminal with pre-computed expected value.
//   kTrunk: Keep all trunk branches. Non-target boundary -> MVS terminal.

namespace open_spiel {

class FullGadgetState;

class FullGadgetGame : public WrappedGame {
 public:
  enum class Mode { kPath, kTrunk };

  // Constructor:
  //   game: original game
  //   trunk_policy: blueprint policy (shared_ptr to keep alive)
  //   resolving_player: which player is being resolved
  //   target_pub_obs: public observation string of the target subgame group
  //   all_boundary_states_by_group: map from pub_obs -> vector of boundary
  //     state history strings
  //   mode: kPath or kTrunk
  //   boundary_portfolios_p0/p1: strategy portfolios for MVS at non-target
  //     boundaries when enumerate_boundary_portfolios is false.
  //   enumerate_boundary_portfolios: if true, lazily enumerate per-state
  //     subtree pure strategies at each boundary state (ignores passed
  //     portfolios). Like MVSGameWithSubtreePureStrategies.
  FullGadgetGame(
      std::shared_ptr<const Game> game,
      std::shared_ptr<const Policy> trunk_policy,
      Player resolving_player,
      const std::string& target_pub_obs,
      const std::unordered_map<std::string, std::vector<std::string>>&
          all_boundary_states_by_group,
      Mode mode,
      std::vector<std::shared_ptr<Policy>> boundary_portfolios_p0 = {},
      std::vector<std::shared_ptr<Policy>> boundary_portfolios_p1 = {},
      bool enumerate_boundary_portfolios = false);

  std::unique_ptr<State> NewInitialState() const override;
  int NumDistinctActions() const override;
  int MaxGameLength() const override;

  Player ResolvingPlayer() const { return resolving_player_; }
  Player NonResolvingPlayer() const { return 1 - resolving_player_; }
  Mode GetMode() const { return mode_; }

 private:
  friend class FullGadgetState;

  // Check if a state history string belongs to the target subgame group
  bool IsTargetBoundaryState(const std::string& history) const;

  // Check if a state history string belongs to any boundary state
  bool IsBoundaryState(const std::string& history) const;

  // Get expected returns for a boundary state (target or non-target)

  // For kPath mode: check if a trunk state can reach the target subgame
  // For kTrunk mode: always returns true
  bool IsOnPath(const std::string& history) const;

  // Get trunk policy action probabilities for a state
  ActionsAndProbs GetTrunkPolicy(const State& state) const;

  // Pre-compute expected returns at a state under trunk policy (recursive)
  std::vector<double> ComputeExpectedReturns(const State& state) const;

  // For kPath: run reachability DFS to find on-path states
  // Returns true if the given state can reach a target boundary state
  bool ComputeOnPathStates(const State& state);

  // Whether MVS is used at non-target boundaries
  bool UseMVSBoundaries() const {
    return enumerate_boundary_portfolios_ ||
           (!boundary_portfolios_p0_.empty() &&
            !boundary_portfolios_p1_.empty());
  }

  // Get payoff for a specific portfolio pair at a boundary state (lazy cached)
  const std::vector<double>& GetBoundaryPayoff(
      const State& state, int p0_idx, int p1_idx,
      const std::vector<std::shared_ptr<Policy>>& port_p0,
      const std::vector<std::shared_ptr<Policy>>& port_p1) const;

  // Compute and cache the full payoff matrix for a boundary state
  void ComputeBoundaryPayoffMatrix(
      const State& state,
      const std::vector<std::shared_ptr<Policy>>& port_p0,
      const std::vector<std::shared_ptr<Policy>>& port_p1) const;

  std::shared_ptr<const Policy> trunk_policy_;
  Player resolving_player_;
  std::string target_pub_obs_;
  Mode mode_;

  // Set of history strings for target boundary states
  std::unordered_set<std::string> target_boundary_states_;

  // Set of ALL boundary state history strings
  std::unordered_set<std::string> all_boundary_states_;

  // Pre-computed expected returns for all boundary states (fallback)
  // For kPath: set of trunk state histories that can reach the target
  // For kTrunk: empty (all trunk states are on path)
  std::unordered_set<std::string> on_path_states_;

  // MVS boundary portfolios (shared across all non-target boundary states)
  // Only used when enumerate_boundary_portfolios_ is false
  std::vector<std::shared_ptr<Policy>> boundary_portfolios_p0_;
  std::vector<std::shared_ptr<Policy>> boundary_portfolios_p1_;

  // If true, enumerate pure strategies per-state at each boundary
  bool enumerate_boundary_portfolios_ = false;

  // Cache: history_key -> flattened matrix [p0*num_p1 + p1] -> {u0, u1}
  mutable std::unordered_map<std::string, std::vector<std::vector<double>>>
      boundary_payoff_cache_;
};

class FullGadgetState : public WrappedState {
 public:
  enum class Phase {
    kTrunk,             // In trunk: resolving player -> chance, non-resolving -> free
    kSubgame,           // In target subgame: both players free
    kBoundaryP0Select,  // Non-target boundary: P0 selects portfolio
    kBoundaryP1Select,  // Non-target boundary: P1 selects portfolio
    kTerminal           // Terminal: matrix payoff, expected value, or game-over
  };

  explicit FullGadgetState(std::shared_ptr<const Game> game,
                           std::unique_ptr<State> initial_state);
  FullGadgetState(const FullGadgetState& other);

  Player CurrentPlayer() const override;
  std::vector<Action> LegalActions() const override;
  std::vector<Action> LegalActions(Player player) const override;
  std::string ActionToString(Player player, Action action_id) const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::vector<double> Rewards() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  std::string ToString() const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;

  Phase GetPhase() const { return phase_; }

 protected:
  void DoApplyAction(Action action_id) override;

 private:
  const FullGadgetGame* GetFullGadgetGame() const;

  // Transition to the appropriate phase based on current state
  void CheckAndTransition();

  // Ensure per-state boundary portfolios are computed (lazy)
  void EnsureBoundaryPortfoliosComputed() const;

  Phase phase_;
  std::vector<double> terminal_returns_;  // Set when transitioning to kTerminal
  Action boundary_p0_choice_ = kInvalidAction;  // MVS boundary selection
  Action boundary_p1_choice_ = kInvalidAction;

  // Per-state lazily computed portfolios (for enumerate_boundary_portfolios)
  mutable std::vector<std::shared_ptr<Policy>> boundary_portfolio_p0_;
  mutable std::vector<std::shared_ptr<Policy>> boundary_portfolio_p1_;
  mutable bool boundary_portfolios_computed_ = false;
};

// Factory function
std::shared_ptr<const FullGadgetGame> CreateFullGadgetGame(
    std::shared_ptr<const Game> game,
    std::shared_ptr<const Policy> trunk_policy,
    Player resolving_player,
    const std::string& target_pub_obs,
    const std::unordered_map<std::string, std::vector<std::string>>&
        all_boundary_states_by_group,
    FullGadgetGame::Mode mode,
    std::vector<std::shared_ptr<Policy>> boundary_portfolios_p0 = {},
    std::vector<std::shared_ptr<Policy>> boundary_portfolios_p1 = {},
    bool enumerate_boundary_portfolios = false);

}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAME_TRANSFORMS_FULL_GADGET_H_
