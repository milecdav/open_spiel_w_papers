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

#include "open_spiel/game_transforms/full_gadget.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/algorithms/expected_returns.h"
#include "open_spiel/game_transforms/matrix_valued_states.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"

namespace open_spiel {

namespace {

const GameType kFullGadgetGameType{
    /*short_name=*/"full_gadget",
    /*long_name=*/"Full Gadget Game",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/false,
    /*parameter_specification=*/{},
    /*default_loadable=*/false};

GameType ConvertType(const GameType& type) {
  GameType new_type = kFullGadgetGameType;
  new_type.long_name = "Full Gadget " + type.long_name;
  new_type.utility = GameType::Utility::kZeroSum;
  new_type.max_num_players = 2;
  new_type.min_num_players = 2;
  new_type.provides_information_state_string =
      type.provides_information_state_string;
  new_type.provides_observation_string = type.provides_observation_string;
  return new_type;
}

}  // namespace

// =============================================================================
// FullGadgetGame implementation
// =============================================================================

FullGadgetGame::FullGadgetGame(
    std::shared_ptr<const Game> game,
    std::shared_ptr<const Policy> trunk_policy,
    Player resolving_player,
    const std::string& target_pub_obs,
    const std::unordered_map<std::string, std::vector<std::string>>&
        all_boundary_states_by_group,
    Mode mode,
    std::vector<std::shared_ptr<Policy>> boundary_portfolios_p0,
    std::vector<std::shared_ptr<Policy>> boundary_portfolios_p1,
    bool enumerate_boundary_portfolios)
    : WrappedGame(game, ConvertType(game->GetType()), game->GetParameters()),
      trunk_policy_(std::move(trunk_policy)),
      resolving_player_(resolving_player),
      target_pub_obs_(target_pub_obs),
      mode_(mode),
      boundary_portfolios_p0_(std::move(boundary_portfolios_p0)),
      boundary_portfolios_p1_(std::move(boundary_portfolios_p1)),
      enumerate_boundary_portfolios_(enumerate_boundary_portfolios) {
  SPIEL_CHECK_GE(resolving_player_, 0);
  SPIEL_CHECK_LT(resolving_player_, 2);
  SPIEL_CHECK_EQ(game->NumPlayers(), 2);

  // Build target boundary states set
  auto it = all_boundary_states_by_group.find(target_pub_obs);
  if (it != all_boundary_states_by_group.end()) {
    for (const auto& hist : it->second) {
      target_boundary_states_.insert(hist);
    }
  }

  // Build all boundary states set
  for (const auto& [pub_obs, states] : all_boundary_states_by_group) {
    for (const auto& hist : states) {
      all_boundary_states_.insert(hist);
    }
  }

  // For kPath mode: run reachability analysis
  if (mode_ == Mode::kPath) {
    auto initial_state = game_->NewInitialState();
    ComputeOnPathStates(*initial_state);
  }
}

bool FullGadgetGame::IsTargetBoundaryState(const std::string& history) const {
  return target_boundary_states_.count(history) > 0;
}

bool FullGadgetGame::IsBoundaryState(const std::string& history) const {
  return all_boundary_states_.count(history) > 0;
}

bool FullGadgetGame::IsOnPath(const std::string& history) const {
  if (mode_ == Mode::kTrunk) return true;
  return on_path_states_.count(history) > 0;
}

ActionsAndProbs FullGadgetGame::GetTrunkPolicy(const State& state) const {
  if (trunk_policy_ != nullptr) {
    auto ap = trunk_policy_->GetStatePolicy(state, resolving_player_);
    if (!ap.empty()) {
      // Filter out zero-probability actions
      ActionsAndProbs filtered;
      for (const auto& [a, p] : ap) {
        if (p > 0.0) filtered.push_back({a, p});
      }
      if (!filtered.empty()) return filtered;
    }
  }
  // Fallback to uniform over legal actions
  auto legal = state.LegalActions();
  if (legal.empty()) return {};
  double p = 1.0 / legal.size();
  ActionsAndProbs result;
  for (Action a : legal) result.push_back({a, p});
  return result;
}

std::vector<double> FullGadgetGame::ComputeExpectedReturns(
    const State& state) const {
  if (state.IsTerminal()) return state.Returns();
  int np = state.NumPlayers();
  std::vector<double> ev(np, 0.0);
  if (state.IsChanceNode()) {
    for (const auto& [a, p] : state.ChanceOutcomes()) {
      auto child = state.Clone();
      child->ApplyAction(a);
      auto child_ev = ComputeExpectedReturns(*child);
      for (int i = 0; i < np; ++i) ev[i] += p * child_ev[i];
    }
  } else {
    Player pl = state.CurrentPlayer();
    ActionsAndProbs ap;
    if (trunk_policy_ != nullptr) {
      ap = trunk_policy_->GetStatePolicy(state, pl);
    }
    if (ap.empty()) {
      // Uniform fallback
      auto legal = state.LegalActions();
      double p = 1.0 / legal.size();
      for (Action a : legal) ap.push_back({a, p});
    }
    for (const auto& [a, p] : ap) {
      if (p <= 0.0) continue;
      auto child = state.Clone();
      child->ApplyAction(a);
      auto child_ev = ComputeExpectedReturns(*child);
      for (int i = 0; i < np; ++i) ev[i] += p * child_ev[i];
    }
  }
  return ev;
}

bool FullGadgetGame::ComputeOnPathStates(const State& state) {
  if (state.IsTerminal()) return false;

  std::string hist = state.HistoryString();

  // Check if this is a boundary state
  if (IsBoundaryState(hist)) {
    bool is_target = IsTargetBoundaryState(hist);
    if (is_target) {
      on_path_states_.insert(hist);
    }
    return is_target;
  }

  // Recurse into children
  bool any_child_on_path = false;
  if (state.IsChanceNode()) {
    for (const auto& [a, p] : state.ChanceOutcomes()) {
      auto child = state.Clone();
      child->ApplyAction(a);
      if (ComputeOnPathStates(*child)) {
        any_child_on_path = true;
      }
    }
  } else {
    for (Action a : state.LegalActions()) {
      auto child = state.Clone();
      child->ApplyAction(a);
      if (ComputeOnPathStates(*child)) {
        any_child_on_path = true;
      }
    }
  }

  if (any_child_on_path) {
    on_path_states_.insert(hist);
  }

  return any_child_on_path;
}

void FullGadgetGame::ComputeBoundaryPayoffMatrix(
    const State& state,
    const std::vector<std::shared_ptr<Policy>>& port_p0,
    const std::vector<std::shared_ptr<Policy>>& port_p1) const {
  std::string key = state.HistoryString();
  int num_p0 = port_p0.size();
  int num_p1 = port_p1.size();

  std::vector<std::vector<double>> matrix;
  matrix.reserve(num_p0 * num_p1);

  for (int i = 0; i < num_p0; ++i) {
    for (int j = 0; j < num_p1; ++j) {
      std::vector<const Policy*> policies = {port_p0[i].get(),
                                              port_p1[j].get()};
      auto returns = algorithms::ExpectedReturns(
          state, policies, /*depth_limit=*/-1,
          /*use_infostate_get_policy=*/false);
      matrix.push_back(returns);
    }
  }

  boundary_payoff_cache_[key] = std::move(matrix);
}

const std::vector<double>& FullGadgetGame::GetBoundaryPayoff(
    const State& state, int p0_idx, int p1_idx,
    const std::vector<std::shared_ptr<Policy>>& port_p0,
    const std::vector<std::shared_ptr<Policy>>& port_p1) const {
  std::string key = state.HistoryString();
  auto it = boundary_payoff_cache_.find(key);
  if (it == boundary_payoff_cache_.end()) {
    ComputeBoundaryPayoffMatrix(state, port_p0, port_p1);
    it = boundary_payoff_cache_.find(key);
  }
  int num_p1 = port_p1.size();
  int idx = p0_idx * num_p1 + p1_idx;
  SPIEL_CHECK_GE(idx, 0);
  SPIEL_CHECK_LT(idx, it->second.size());
  return it->second[idx];
}

std::unique_ptr<State> FullGadgetGame::NewInitialState() const {
  return std::make_unique<FullGadgetState>(
      shared_from_this(), game_->NewInitialState());
}

int FullGadgetGame::NumDistinctActions() const {
  int base = game_->NumDistinctActions();
  if (!enumerate_boundary_portfolios_) {
    // With fixed portfolios, we know the sizes
    base = std::max(base, static_cast<int>(boundary_portfolios_p0_.size()));
    base = std::max(base, static_cast<int>(boundary_portfolios_p1_.size()));
  }
  // For enumerate mode, per-state portfolios may vary; use a generous bound
  // The actual legal actions are determined by LegalActions() at each state
  return std::max(base, 100);  // Upper bound for portfolio actions
}

int FullGadgetGame::MaxGameLength() const {
  // +2 for the two portfolio selection actions at boundary
  return game_->MaxGameLength() + (UseMVSBoundaries() ? 2 : 0);
}

// =============================================================================
// FullGadgetState implementation
// =============================================================================

FullGadgetState::FullGadgetState(std::shared_ptr<const Game> game,
                                 std::unique_ptr<State> initial_state)
    : WrappedState(game, std::move(initial_state)),
      phase_(Phase::kTrunk) {
  // Check if we need to transition right at the start
  CheckAndTransition();
}

FullGadgetState::FullGadgetState(const FullGadgetState& other)
    : WrappedState(other.game_,
                   other.state_ ? other.state_->Clone() : nullptr),
      phase_(other.phase_),
      terminal_returns_(other.terminal_returns_),
      boundary_p0_choice_(other.boundary_p0_choice_),
      boundary_p1_choice_(other.boundary_p1_choice_),
      boundary_portfolio_p0_(other.boundary_portfolio_p0_),
      boundary_portfolio_p1_(other.boundary_portfolio_p1_),
      boundary_portfolios_computed_(other.boundary_portfolios_computed_) {
  history_ = other.history_;
}

void FullGadgetState::EnsureBoundaryPortfoliosComputed() const {
  if (boundary_portfolios_computed_) return;
  const auto* fg_game = GetFullGadgetGame();
  if (fg_game->enumerate_boundary_portfolios_) {
    boundary_portfolio_p0_ = EnumerateSubtreePureStrategies(*state_, 0);
    boundary_portfolio_p1_ = EnumerateSubtreePureStrategies(*state_, 1);
  } else {
    boundary_portfolio_p0_ = fg_game->boundary_portfolios_p0_;
    boundary_portfolio_p1_ = fg_game->boundary_portfolios_p1_;
  }
  boundary_portfolios_computed_ = true;
}

const FullGadgetGame* FullGadgetState::GetFullGadgetGame() const {
  return down_cast<const FullGadgetGame*>(game_.get());
}

void FullGadgetState::CheckAndTransition() {
  const auto* fg_game = GetFullGadgetGame();

  if (phase_ == Phase::kTrunk) {
    // Check if underlying state is terminal (game ended in trunk)
    if (state_->IsTerminal()) {
      phase_ = Phase::kTerminal;
      terminal_returns_ = state_->Returns();
      return;
    }

    // Check if we've reached a boundary state
    std::string hist = state_->HistoryString();
    if (fg_game->IsTargetBoundaryState(hist)) {
      phase_ = Phase::kSubgame;
      return;
    }
    if (fg_game->IsBoundaryState(hist)) {
      if (fg_game->UseMVSBoundaries()) {
        phase_ = (fg_game->ResolvingPlayer() == 0)
                     ? Phase::kBoundaryP0Select
                     : Phase::kBoundaryP1Select;
      } else {
        phase_ = Phase::kTerminal;
        SpielFatalError(
            "FullGadget requires portfolio-based non-target boundaries. "
            "Provide boundary portfolios or enable enumerate_boundary_portfolios.");
      }
      return;
    }

    // For kPath mode: check if this state is on the path to the target
    if (fg_game->GetMode() == FullGadgetGame::Mode::kPath &&
        !fg_game->IsOnPath(hist)) {
      phase_ = Phase::kTerminal;
      terminal_returns_ = fg_game->ComputeExpectedReturns(*state_);
      return;
    }

  } else if (phase_ == Phase::kSubgame) {
    if (state_->IsTerminal()) {
      phase_ = Phase::kTerminal;
      terminal_returns_ = state_->Returns();
      return;
    }
  }
}

Player FullGadgetState::CurrentPlayer() const {
  const auto* fg_game = GetFullGadgetGame();

  if (phase_ == Phase::kTerminal) {
    return kTerminalPlayerId;
  }

  if (phase_ == Phase::kBoundaryP0Select) return 0;
  if (phase_ == Phase::kBoundaryP1Select) return 1;

  if (phase_ == Phase::kSubgame) {
    return state_->CurrentPlayer();
  }

  // In trunk phase:
  if (state_->IsTerminal()) {
    return kTerminalPlayerId;
  }
  if (state_->IsChanceNode()) {
    return kChancePlayerId;
  }
  // Resolving player's decisions become chance in trunk
  if (state_->CurrentPlayer() == fg_game->ResolvingPlayer()) {
    return kChancePlayerId;
  }
  // Non-resolving player acts freely
  return state_->CurrentPlayer();
}

std::vector<Action> FullGadgetState::LegalActions() const {
  if (phase_ == Phase::kTerminal) return {};

  if (phase_ == Phase::kBoundaryP0Select) {
    EnsureBoundaryPortfoliosComputed();
    std::vector<Action> actions;
    for (int i = 0; i < static_cast<int>(boundary_portfolio_p0_.size()); ++i)
      actions.push_back(i);
    return actions;
  }
  if (phase_ == Phase::kBoundaryP1Select) {
    EnsureBoundaryPortfoliosComputed();
    std::vector<Action> actions;
    for (int i = 0; i < static_cast<int>(boundary_portfolio_p1_.size()); ++i)
      actions.push_back(i);
    return actions;
  }

  if (phase_ == Phase::kSubgame) return state_->LegalActions();

  // Trunk phase
  return state_->LegalActions();
}

std::vector<Action> FullGadgetState::LegalActions(Player player) const {
  if (player == CurrentPlayer()) {
    return LegalActions();
  }
  return {};
}

std::string FullGadgetState::ActionToString(Player player,
                                             Action action_id) const {
  if (phase_ == Phase::kSubgame || phase_ == Phase::kTrunk) {
    return state_->ActionToString(player, action_id);
  }
  if (phase_ == Phase::kBoundaryP0Select ||
      phase_ == Phase::kBoundaryP1Select) {
    return absl::StrCat("boundary_portfolio_", action_id);
  }
  return absl::StrCat("action_", action_id);
}

bool FullGadgetState::IsTerminal() const {
  return phase_ == Phase::kTerminal;
}

std::vector<double> FullGadgetState::Returns() const {
  if (phase_ != Phase::kTerminal) {
    SpielFatalError("Returns() called on non-terminal FullGadgetState");
  }
  return terminal_returns_;
}

std::vector<double> FullGadgetState::Rewards() const {
  if (IsTerminal()) {
    return Returns();
  }
  return std::vector<double>(2, 0.0);
}

std::string FullGadgetState::InformationStateString(Player player) const {
  const auto* fg_game = GetFullGadgetGame();

  if (phase_ == Phase::kTerminal) {
    return absl::StrCat("full_gadget:terminal:", player);
  }

  if (phase_ == Phase::kBoundaryP0Select ||
      phase_ == Phase::kBoundaryP1Select) {
    // Simultaneous game: each player sees their own info state + player-specific
    // suffix so P1 cannot observe P0's choice
    return absl::StrCat("full_boundary:",
                        state_->InformationStateString(player),
                        ":BSEL", player);
  }

  if (phase_ == Phase::kSubgame) {
    return absl::StrCat("full_F:subgame:",
                        state_->InformationStateString(player));
  }

  // Trunk phase
  if (player == fg_game->ResolvingPlayer()) {
    return "full_trunk:resolving";
  } else {
    return absl::StrCat("full_trunk:", state_->InformationStateString(player));
  }
}

std::string FullGadgetState::ObservationString(Player player) const {
  return InformationStateString(player);
}

std::string FullGadgetState::ToString() const {
  std::string result = "FullGadgetState(";
  switch (phase_) {
    case Phase::kTrunk:
      result += absl::StrCat("phase=Trunk, state=", state_->ToString());
      break;
    case Phase::kSubgame:
      result += absl::StrCat("phase=Subgame, state=", state_->ToString());
      break;
    case Phase::kBoundaryP0Select:
      result += absl::StrCat("phase=BoundaryP0Select, state=",
                              state_->ToString());
      break;
    case Phase::kBoundaryP1Select:
      result += absl::StrCat("phase=BoundaryP1Select, p0_choice=",
                              boundary_p0_choice_, ", state=",
                              state_->ToString());
      break;
    case Phase::kTerminal:
      result += "phase=Terminal";
      break;
  }
  result += ")";
  return result;
}

std::unique_ptr<State> FullGadgetState::Clone() const {
  return std::make_unique<FullGadgetState>(*this);
}

std::vector<std::pair<Action, double>> FullGadgetState::ChanceOutcomes() const {
  const auto* fg_game = GetFullGadgetGame();

  if (phase_ == Phase::kBoundaryP0Select ||
      phase_ == Phase::kBoundaryP1Select) {
    return {};  // Player decision nodes, not chance
  }

  if (phase_ == Phase::kSubgame) {
    if (state_->IsChanceNode()) {
      return state_->ChanceOutcomes();
    }
    return {};
  }

  if (phase_ == Phase::kTrunk) {
    if (state_->IsChanceNode()) {
      return state_->ChanceOutcomes();
    }
    // Resolving player's turn -> return trunk policy as chance outcomes
    if (state_->CurrentPlayer() == fg_game->ResolvingPlayer()) {
      return fg_game->GetTrunkPolicy(*state_);
    }
  }

  return {};
}

void FullGadgetState::DoApplyAction(Action action_id) {
  if (phase_ == Phase::kTerminal) {
    SpielFatalError("Cannot apply action to terminal FullGadgetState");
  }

  const auto* fg_game = GetFullGadgetGame();

  // Handle MVS boundary portfolio selection (don't touch underlying state)
  if (phase_ == Phase::kBoundaryP0Select) {
    EnsureBoundaryPortfoliosComputed();
    boundary_p0_choice_ = action_id;
    if (boundary_p1_choice_ == kInvalidAction) {
      phase_ = Phase::kBoundaryP1Select;
    } else {
      phase_ = Phase::kTerminal;
      terminal_returns_ = fg_game->GetBoundaryPayoff(
          *state_, boundary_p0_choice_, boundary_p1_choice_,
          boundary_portfolio_p0_, boundary_portfolio_p1_);
    }
    return;
  }
  if (phase_ == Phase::kBoundaryP1Select) {
    EnsureBoundaryPortfoliosComputed();
    boundary_p1_choice_ = action_id;
    if (boundary_p0_choice_ == kInvalidAction) {
      phase_ = Phase::kBoundaryP0Select;
    } else {
      phase_ = Phase::kTerminal;
      terminal_returns_ = fg_game->GetBoundaryPayoff(
          *state_, boundary_p0_choice_, boundary_p1_choice_,
          boundary_portfolio_p0_, boundary_portfolio_p1_);
    }
    return;
  }

  // Apply the action to the underlying state
  state_->ApplyAction(action_id);

  // Check if we need to transition phases
  CheckAndTransition();
}

// =============================================================================
// Factory function
// =============================================================================

std::shared_ptr<const FullGadgetGame> CreateFullGadgetGame(
    std::shared_ptr<const Game> game,
    std::shared_ptr<const Policy> trunk_policy,
    Player resolving_player,
    const std::string& target_pub_obs,
    const std::unordered_map<std::string, std::vector<std::string>>&
        all_boundary_states_by_group,
    FullGadgetGame::Mode mode,
    std::vector<std::shared_ptr<Policy>> boundary_portfolios_p0,
    std::vector<std::shared_ptr<Policy>> boundary_portfolios_p1,
    bool enumerate_boundary_portfolios) {
  return std::make_shared<FullGadgetGame>(
      game, std::move(trunk_policy), resolving_player, target_pub_obs,
      all_boundary_states_by_group, mode,
      std::move(boundary_portfolios_p0), std::move(boundary_portfolios_p1),
      enumerate_boundary_portfolios);
}

}  // namespace open_spiel
