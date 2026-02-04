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

#ifndef OPEN_SPIEL_GAME_TRANSFORMS_MATRIX_VALUED_STATES_H_
#define OPEN_SPIEL_GAME_TRANSFORMS_MATRIX_VALUED_STATES_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "open_spiel/game_transforms/game_wrapper.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"

// This transformation creates a matrix-valued depth-limited game as defined in
// Definition 2 of the paper "Adapting Beyond the Depth Limit" (arXiv:2501.10464).
//
// At depth limit d, the game transforms: both players simultaneously select
// from strategy portfolios P1 and P2, receiving payoff
// E[u_i(z) | z ~ (pi_1, pi_2), z extends h].
//
// The simultaneous portfolio selection is converted to turn-based:
// - Player 0 selects portfolio index first (but P1 doesn't observe this)
// - Player 1 selects portfolio index second
// - Game terminates with matrix payoff
//
// Usage:
//   auto game = LoadGame("kuhn_poker");
//   auto uniform = std::make_shared<UniformPolicy>();
//   std::vector<std::shared_ptr<Policy>> p1_portfolios = {uniform};
//   std::vector<std::shared_ptr<Policy>> p2_portfolios = {uniform};
//   auto mvs_game = std::make_shared<MVSGame>(
//       game, p1_portfolios, p2_portfolios, /*depth_limit=*/2);

namespace open_spiel {

class MVSState;

class MVSGame : public WrappedGame {
 public:
  // How to measure depth for the depth limit.
  enum class DepthMode {
    kActionBased,  // Count non-chance player decisions
    kRoundBased    // Count chance nodes (card deals) as round boundaries
  };

  MVSGame(std::shared_ptr<const Game> game,
          std::vector<std::shared_ptr<Policy>> portfolios_p1,
          std::vector<std::shared_ptr<Policy>> portfolios_p2,
          int depth_limit,
          DepthMode depth_mode = DepthMode::kActionBased);

  std::unique_ptr<State> NewInitialState() const override;

  // Max of original actions and portfolio sizes (portfolios use action indices)
  int NumDistinctActions() const override;

  // Depth limit + 2 (for the two portfolio selection actions)
  int MaxGameLength() const override;

  // Access to configuration
  int DepthLimit() const { return depth_limit_; }
  DepthMode GetDepthMode() const { return depth_mode_; }
  int NumPortfoliosP1() const { return portfolios_p1_.size(); }
  int NumPortfoliosP2() const { return portfolios_p2_.size(); }

  const std::vector<std::shared_ptr<Policy>>& PortfoliosP1() const {
    return portfolios_p1_;
  }
  const std::vector<std::shared_ptr<Policy>>& PortfoliosP2() const {
    return portfolios_p2_;
  }

  // Get the payoff for a specific portfolio pair at a given history.
  // history_key is the string identifying the state where depth limit was hit.
  // Lazily computes and caches payoff matrices.
  const std::vector<double>& GetPayoff(const State& state,
                                        int p1_idx, int p2_idx) const;

 private:
  friend class MVSState;

  // Compute the full payoff matrix for a given state and cache it.
  void ComputePayoffMatrix(const State& state) const;

  std::vector<std::shared_ptr<Policy>> portfolios_p1_;
  std::vector<std::shared_ptr<Policy>> portfolios_p2_;
  int depth_limit_;
  DepthMode depth_mode_;

  // Cache: history_key -> flattened matrix [p1*P2_size + p2] -> {u1, u2, ...}
  mutable std::unordered_map<std::string, std::vector<std::vector<double>>>
      payoff_cache_;
};

class MVSState : public WrappedState {
 public:
  // Phase tracking for the state machine
  enum class Phase {
    kNormal,         // Before depth limit, normal game play
    kPortfolioP1,    // P1 selecting portfolio (simultaneous, turn-based)
    kPortfolioP2,    // P2 selecting portfolio
    kMatrixTerminal  // Terminal with matrix payoff
  };

  MVSState(std::shared_ptr<const Game> game, std::unique_ptr<State> state);
  MVSState(const MVSState& other);

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

  // Access to phase for debugging/testing
  Phase GetPhase() const { return phase_; }
  int CurrentDepth() const { return current_depth_; }

 protected:
  void DoApplyAction(Action action_id) override;

 private:
  const MVSGame* GetMVSGame() const;

  // Check if we're at the depth limit and should transition to portfolio selection
  bool AtDepthLimit() const;

  // Count the current round (for round-based depth mode)
  int ComputeCurrentRound() const;

  Phase phase_ = Phase::kNormal;
  int current_depth_ = 0;
  int current_round_ = 0;  // For round-based depth mode
  Action p1_choice_ = kInvalidAction;
  Action p2_choice_ = kInvalidAction;
};

// Helper function to create an MVS game with explicit portfolios
std::shared_ptr<const MVSGame> CreateMVSGame(
    std::shared_ptr<const Game> game,
    std::vector<std::shared_ptr<Policy>> portfolios_p1,
    std::vector<std::shared_ptr<Policy>> portfolios_p2,
    int depth_limit,
    MVSGame::DepthMode depth_mode = MVSGame::DepthMode::kActionBased);

// ============================================================================
// MVS Game with Automatic Subtree Pure Strategy Enumeration
// ============================================================================

// This variant of MVS game automatically enumerates pure strategies
// at each depth-limited node, rather than using a fixed portfolio.
// This is more memory-efficient for larger games since strategies are
// computed lazily per-subtree rather than for the entire game.

class MVSGameWithSubtreePureStrategies;

class MVSStateWithSubtreePureStrategies : public WrappedState {
 public:
  using Phase = MVSState::Phase;

  MVSStateWithSubtreePureStrategies(std::shared_ptr<const Game> game,
                                     std::unique_ptr<State> state);
  MVSStateWithSubtreePureStrategies(const MVSStateWithSubtreePureStrategies& other);

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
  int CurrentDepth() const { return current_depth_; }

  // Get the portfolios computed for this state (lazily computed)
  const std::vector<std::shared_ptr<Policy>>& GetPortfolioP0() const;
  const std::vector<std::shared_ptr<Policy>>& GetPortfolioP1() const;

 protected:
  void DoApplyAction(Action action_id) override;

 private:
  const MVSGameWithSubtreePureStrategies* GetMVSGame() const;
  bool AtDepthLimit() const;
  void EnsurePortfoliosComputed() const;

  Phase phase_ = Phase::kNormal;
  int current_depth_ = 0;
  int current_round_ = 0;
  Action p1_choice_ = kInvalidAction;
  Action p2_choice_ = kInvalidAction;

  // Lazily computed portfolios for this depth-limited state
  mutable std::vector<std::shared_ptr<Policy>> portfolio_p0_;
  mutable std::vector<std::shared_ptr<Policy>> portfolio_p1_;
  mutable bool portfolios_computed_ = false;
};

class MVSGameWithSubtreePureStrategies : public WrappedGame {
 public:
  using DepthMode = MVSGame::DepthMode;

  MVSGameWithSubtreePureStrategies(std::shared_ptr<const Game> game,
                                    int depth_limit,
                                    DepthMode depth_mode = DepthMode::kActionBased);

  std::unique_ptr<State> NewInitialState() const override;
  int NumDistinctActions() const override;
  int MaxGameLength() const override;

  int DepthLimit() const { return depth_limit_; }
  DepthMode GetDepthMode() const { return depth_mode_; }

  // Get the payoff for a specific portfolio pair at a given state.
  // Uses the state's lazily-computed portfolios.
  const std::vector<double>& GetPayoff(
      const MVSStateWithSubtreePureStrategies& mvs_state,
      int p1_idx, int p2_idx) const;

 private:
  friend class MVSStateWithSubtreePureStrategies;

  int depth_limit_;
  DepthMode depth_mode_;

  // Cache: (history_key, p1_idx, p2_idx) -> returns
  // Using nested structure for simplicity
  mutable std::unordered_map<std::string,
      std::unordered_map<int, std::vector<double>>> payoff_cache_;
};

// Factory function for MVS game with automatic subtree pure strategies
std::shared_ptr<const MVSGameWithSubtreePureStrategies>
CreateMVSGameWithSubtreePureStrategies(
    std::shared_ptr<const Game> game,
    int depth_limit,
    MVSGame::DepthMode depth_mode = MVSGame::DepthMode::kActionBased);

// ============================================================================
// Pure Strategy Enumeration Utilities
// ============================================================================

// A pure strategy for a single player in a subtree.
// Maps information state strings to the chosen action at that infostate.
using PureStrategyMap = std::unordered_map<std::string, Action>;

// A policy that implements a pure strategy defined by a PureStrategyMap.
// Falls back to uniform random for infostates not in the map.
class SubtreePureStrategy : public Policy {
 public:
  SubtreePureStrategy(PureStrategyMap strategy, Player player);

  ActionsAndProbs GetStatePolicy(const State& state,
                                  Player player) const override;
  ActionsAndProbs GetStatePolicy(const std::string& info_state) const override;

 private:
  PureStrategyMap strategy_;
  Player player_;
};

// Information about infostates in a subtree for a single player.
struct SubtreeInfostates {
  // List of information state strings reachable in the subtree.
  std::vector<std::string> infostates;
  // For each infostate, the list of legal actions.
  std::unordered_map<std::string, std::vector<Action>> legal_actions;
};

// Collect all information states reachable from the given state for a player.
// Only traverses the subtree rooted at 'state'.
SubtreeInfostates CollectSubtreeInfostates(const State& state, Player player);

// Enumerate all pure strategies for a player in the subtree rooted at 'state'.
// Returns a vector of PureStrategyMaps, one for each pure strategy.
// WARNING: This is exponential in the number of infostates!
std::vector<PureStrategyMap> EnumeratePureStrategies(
    const SubtreeInfostates& infostates);

// Convert pure strategy maps to Policy objects for use with MVSGame.
std::vector<std::shared_ptr<Policy>> ConvertToPortfolio(
    const std::vector<PureStrategyMap>& pure_strategies, Player player);

// Convenience function: enumerate all pure strategies for a player from a state
// and return them as a portfolio of policies.
std::vector<std::shared_ptr<Policy>> EnumerateSubtreePureStrategies(
    const State& state, Player player);

}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAME_TRANSFORMS_MATRIX_VALUED_STATES_H_
