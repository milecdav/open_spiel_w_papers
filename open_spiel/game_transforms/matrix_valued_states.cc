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

#include "open_spiel/game_transforms/matrix_valued_states.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/algorithms/expected_returns.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"

namespace open_spiel {

namespace {

// Game type for the MVS transformation
const GameType kGameType{
    /*short_name=*/"matrix_valued_states",
    /*long_name=*/"Matrix-Valued States Depth-Limited Game",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kSampledStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kGeneralSum,
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
  GameType new_type = kGameType;
  new_type.long_name = "Matrix-Valued States " + type.long_name;
  new_type.chance_mode = type.chance_mode;
  new_type.utility = type.utility;
  new_type.reward_model = GameType::RewardModel::kTerminal;
  new_type.max_num_players = type.max_num_players;
  new_type.min_num_players = type.min_num_players;
  new_type.provides_information_state_string =
      type.provides_information_state_string;
  new_type.provides_observation_string = type.provides_observation_string;
  return new_type;
}

}  // namespace

// ============================================================================
// MVSGame implementation
// ============================================================================

MVSGame::MVSGame(std::shared_ptr<const Game> game,
                 std::vector<std::shared_ptr<Policy>> portfolios_p1,
                 std::vector<std::shared_ptr<Policy>> portfolios_p2,
                 int depth_limit,
                 DepthMode depth_mode)
    : WrappedGame(game, ConvertType(game->GetType()), game->GetParameters()),
      portfolios_p1_(std::move(portfolios_p1)),
      portfolios_p2_(std::move(portfolios_p2)),
      depth_limit_(depth_limit),
      depth_mode_(depth_mode) {
  SPIEL_CHECK_GT(portfolios_p1_.size(), 0);
  SPIEL_CHECK_GT(portfolios_p2_.size(), 0);
  SPIEL_CHECK_GE(depth_limit_, 0);
  SPIEL_CHECK_EQ(game->NumPlayers(), 2);
}

std::unique_ptr<State> MVSGame::NewInitialState() const {
  return std::make_unique<MVSState>(shared_from_this(),
                                     game_->NewInitialState());
}

int MVSGame::NumDistinctActions() const {
  // Need to accommodate both original game actions and portfolio indices
  return std::max({game_->NumDistinctActions(),
                   static_cast<int>(portfolios_p1_.size()),
                   static_cast<int>(portfolios_p2_.size())});
}

int MVSGame::MaxGameLength() const {
  // Original game length (capped at depth limit) + 2 portfolio selections
  return depth_limit_ + 2;
}

const std::vector<double>& MVSGame::GetPayoff(const State& state,
                                               int p1_idx, int p2_idx) const {
  std::string history_key = state.HistoryString();
  auto it = payoff_cache_.find(history_key);
  if (it == payoff_cache_.end()) {
    ComputePayoffMatrix(state);
    it = payoff_cache_.find(history_key);
  }

  int idx = p1_idx * portfolios_p2_.size() + p2_idx;
  SPIEL_CHECK_LT(idx, it->second.size());
  return it->second[idx];
}

void MVSGame::ComputePayoffMatrix(const State& state) const {
  std::string history_key = state.HistoryString();

  std::vector<std::vector<double>> matrix;
  matrix.reserve(portfolios_p1_.size() * portfolios_p2_.size());

  for (size_t i = 0; i < portfolios_p1_.size(); ++i) {
    for (size_t j = 0; j < portfolios_p2_.size(); ++j) {
      std::vector<const Policy*> policies = {portfolios_p1_[i].get(),
                                              portfolios_p2_[j].get()};
      // Use ExpectedReturns with unlimited depth from this state
      auto returns = algorithms::ExpectedReturns(
          state, policies, /*depth_limit=*/-1, /*use_infostate_get_policy=*/false);
      matrix.push_back(returns);
    }
  }

  payoff_cache_[history_key] = std::move(matrix);
}

// ============================================================================
// MVSState implementation
// ============================================================================

MVSState::MVSState(std::shared_ptr<const Game> game,
                   std::unique_ptr<State> state)
    : WrappedState(game, std::move(state)),
      phase_(Phase::kNormal),
      current_depth_(0),
      current_round_(0),
      p1_choice_(kInvalidAction),
      p2_choice_(kInvalidAction) {}

MVSState::MVSState(const MVSState& other)
    : WrappedState(other),
      phase_(other.phase_),
      current_depth_(other.current_depth_),
      current_round_(other.current_round_),
      p1_choice_(other.p1_choice_),
      p2_choice_(other.p2_choice_) {}

const MVSGame* MVSState::GetMVSGame() const {
  return down_cast<const MVSGame*>(game_.get());
}

bool MVSState::AtDepthLimit() const {
  // Don't trigger depth limit if already terminal or at a chance node
  if (state_->IsTerminal() || state_->IsChanceNode()) {
    return false;
  }

  const MVSGame* game = GetMVSGame();
  if (game->GetDepthMode() == MVSGame::DepthMode::kActionBased) {
    return current_depth_ >= game->DepthLimit();
  } else {
    // Round-based: compare current round count
    return current_round_ >= game->DepthLimit();
  }
}

int MVSState::ComputeCurrentRound() const {
  // For round-based depth mode: count completed rounds
  // A round is completed when a chance node deals cards
  return current_round_;
}

Player MVSState::CurrentPlayer() const {
  switch (phase_) {
    case Phase::kNormal:
      if (AtDepthLimit()) {
        return 0;  // P1 selects first
      }
      return state_->CurrentPlayer();
    case Phase::kPortfolioP1:
      return 0;
    case Phase::kPortfolioP2:
      return 1;
    case Phase::kMatrixTerminal:
      return kTerminalPlayerId;
  }
  SpielFatalError("Unknown phase in MVSState::CurrentPlayer");
}

std::vector<Action> MVSState::LegalActions() const {
  const MVSGame* game = GetMVSGame();

  switch (phase_) {
    case Phase::kNormal:
      if (AtDepthLimit()) {
        // P1 portfolio selection
        std::vector<Action> actions;
        for (int i = 0; i < game->NumPortfoliosP1(); ++i) {
          actions.push_back(i);
        }
        return actions;
      }
      return state_->LegalActions();
    case Phase::kPortfolioP1: {
      std::vector<Action> actions;
      for (int i = 0; i < game->NumPortfoliosP1(); ++i) {
        actions.push_back(i);
      }
      return actions;
    }
    case Phase::kPortfolioP2: {
      std::vector<Action> actions;
      for (int i = 0; i < game->NumPortfoliosP2(); ++i) {
        actions.push_back(i);
      }
      return actions;
    }
    case Phase::kMatrixTerminal:
      return {};
  }
  SpielFatalError("Unknown phase in MVSState::LegalActions");
}

std::vector<Action> MVSState::LegalActions(Player player) const {
  // In a sequential game, only the current player has legal actions
  if (player != CurrentPlayer()) {
    return {};
  }
  return LegalActions();
}

std::string MVSState::ActionToString(Player player, Action action_id) const {
  switch (phase_) {
    case Phase::kNormal:
      if (AtDepthLimit()) {
        return absl::StrCat("P", player, "_portfolio_", action_id);
      }
      return state_->ActionToString(player, action_id);
    case Phase::kPortfolioP1:
    case Phase::kPortfolioP2:
      return absl::StrCat("P", player, "_portfolio_", action_id);
    case Phase::kMatrixTerminal:
      return "terminal";
  }
  SpielFatalError("Unknown phase in MVSState::ActionToString");
}

bool MVSState::IsTerminal() const {
  if (phase_ == Phase::kMatrixTerminal) {
    return true;
  }
  if (phase_ == Phase::kNormal && state_->IsTerminal()) {
    return true;
  }
  return false;
}

std::vector<double> MVSState::Returns() const {
  if (phase_ == Phase::kMatrixTerminal) {
    const MVSGame* game = GetMVSGame();
    return game->GetPayoff(*state_, p1_choice_, p2_choice_);
  }
  // If the underlying game terminated before depth limit
  if (state_->IsTerminal()) {
    return state_->Returns();
  }
  return std::vector<double>(num_players_, 0.0);
}

std::vector<double> MVSState::Rewards() const {
  // This is a terminal reward model - rewards only at terminal
  if (IsTerminal()) {
    return Returns();
  }
  return std::vector<double>(num_players_, 0.0);
}

std::string MVSState::InformationStateString(Player player) const {
  std::string base = state_->InformationStateString(player);

  switch (phase_) {
    case Phase::kNormal:
      if (AtDepthLimit()) {
        // Transitioning to portfolio selection
        return absl::StrCat(base, ":MVS_SELECT");
      }
      return base;
    case Phase::kPortfolioP1:
    case Phase::kPortfolioP2:
      // Simultaneous selection: neither player knows other's choice yet
      return absl::StrCat(base, ":MVS_SELECT");
    case Phase::kMatrixTerminal:
      // At terminal, include own choice for proper value propagation
      if (player == 0) {
        return absl::StrCat(base, ":MVS:", p1_choice_);
      } else {
        return absl::StrCat(base, ":MVS:", p2_choice_);
      }
  }
  SpielFatalError("Unknown phase in MVSState::InformationStateString");
}

std::string MVSState::ObservationString(Player player) const {
  // Use same logic as information state for observations
  return InformationStateString(player);
}

std::string MVSState::ToString() const {
  std::string result = state_->ToString();
  switch (phase_) {
    case Phase::kNormal:
      if (AtDepthLimit()) {
        absl::StrAppend(&result, "\n[MVS: At depth limit, P1 selecting]");
      }
      break;
    case Phase::kPortfolioP1:
      absl::StrAppend(&result, "\n[MVS: P1 selecting portfolio]");
      break;
    case Phase::kPortfolioP2:
      absl::StrAppend(&result, "\n[MVS: P2 selecting portfolio, P1 chose ",
                      p1_choice_, "]");
      break;
    case Phase::kMatrixTerminal:
      absl::StrAppend(&result, "\n[MVS: Terminal, P1=", p1_choice_, ", P2=",
                      p2_choice_, "]");
      break;
  }
  return result;
}

std::unique_ptr<State> MVSState::Clone() const {
  return std::make_unique<MVSState>(*this);
}

std::vector<std::pair<Action, double>> MVSState::ChanceOutcomes() const {
  if (phase_ != Phase::kNormal || AtDepthLimit()) {
    return {};
  }
  return state_->ChanceOutcomes();
}

void MVSState::DoApplyAction(Action action_id) {
  switch (phase_) {
    case Phase::kNormal:
      if (AtDepthLimit()) {
        // This is P1's portfolio selection
        p1_choice_ = action_id;
        phase_ = Phase::kPortfolioP2;
      } else {
        // Normal game action
        bool was_chance = state_->IsChanceNode();
        state_->ApplyAction(action_id);

        // Update depth counters
        if (!was_chance) {
          current_depth_++;
        } else if (GetMVSGame()->GetDepthMode() ==
                   MVSGame::DepthMode::kRoundBased) {
          // Increment round counter after chance node
          current_round_++;
        }

        // Check if we should transition after this action
        if (!state_->IsTerminal() && !state_->IsChanceNode() && AtDepthLimit()) {
          phase_ = Phase::kPortfolioP1;
        }
      }
      break;

    case Phase::kPortfolioP1:
      p1_choice_ = action_id;
      phase_ = Phase::kPortfolioP2;
      break;

    case Phase::kPortfolioP2:
      p2_choice_ = action_id;
      phase_ = Phase::kMatrixTerminal;
      break;

    case Phase::kMatrixTerminal:
      SpielFatalError("Cannot apply action to terminal state");
  }
}

std::shared_ptr<const MVSGame> CreateMVSGame(
    std::shared_ptr<const Game> game,
    std::vector<std::shared_ptr<Policy>> portfolios_p1,
    std::vector<std::shared_ptr<Policy>> portfolios_p2,
    int depth_limit,
    MVSGame::DepthMode depth_mode) {
  return std::make_shared<MVSGame>(std::move(game), std::move(portfolios_p1),
                                   std::move(portfolios_p2), depth_limit,
                                   depth_mode);
}

}  // namespace open_spiel
