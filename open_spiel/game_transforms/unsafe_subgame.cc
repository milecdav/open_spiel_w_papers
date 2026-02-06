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

#include "open_spiel/game_transforms/unsafe_subgame.h"

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {

// =============================================================================
// UnsafeSubgameGame implementation
// =============================================================================

UnsafeSubgameGame::UnsafeSubgameGame(
    std::shared_ptr<const Game> game,
    std::vector<std::unique_ptr<State>> roots,
    std::vector<double> reach_probs)
    : WrappedGame(game, game->GetType(), game->GetParameters()),
      roots_(std::move(roots)),
      reach_probs_(std::move(reach_probs)),
      k_(0.0) {
  SPIEL_CHECK_EQ(roots_.size(), reach_probs_.size());
  SPIEL_CHECK_GT(roots_.size(), 0);

  // Compute normalization constant
  for (double p : reach_probs_) {
    SPIEL_CHECK_GE(p, 0.0);
    k_ += p;
  }
  SPIEL_CHECK_GT(k_, 0.0);
}

std::unique_ptr<State> UnsafeSubgameGame::NewInitialState() const {
  return std::make_unique<UnsafeSubgameState>(shared_from_this());
}

std::unique_ptr<State> UnsafeSubgameGame::CloneRootState(int idx) const {
  SPIEL_CHECK_GE(idx, 0);
  SPIEL_CHECK_LT(idx, roots_.size());
  return roots_[idx]->Clone();
}

// =============================================================================
// UnsafeSubgameState implementation
// =============================================================================

UnsafeSubgameState::UnsafeSubgameState(std::shared_ptr<const Game> game)
    : WrappedState(game, nullptr),
      phase_(Phase::kChance),
      selected_root_(-1) {}

UnsafeSubgameState::UnsafeSubgameState(const UnsafeSubgameState& other)
    : WrappedState(other.game_, other.state_ ? other.state_->Clone() : nullptr),
      phase_(other.phase_),
      selected_root_(other.selected_root_) {
  history_ = other.history_;
}

const UnsafeSubgameGame* UnsafeSubgameState::GetUnsafeSubgameGame() const {
  return down_cast<const UnsafeSubgameGame*>(game_.get());
}

Player UnsafeSubgameState::CurrentPlayer() const {
  if (phase_ == Phase::kChance) {
    return kChancePlayerId;
  }
  return state_->CurrentPlayer();
}

std::vector<Action> UnsafeSubgameState::LegalActions() const {
  if (phase_ == Phase::kChance) {
    std::vector<Action> actions;
    const UnsafeSubgameGame* g = GetUnsafeSubgameGame();
    for (int i = 0; i < g->NumSubgameRoots(); ++i) {
      actions.push_back(i);
    }
    return actions;
  }
  return state_->LegalActions();
}

std::string UnsafeSubgameState::ActionToString(Player player,
                                                Action action) const {
  if (phase_ == Phase::kChance) {
    return absl::StrCat("root_", action);
  }
  return state_->ActionToString(player, action);
}

bool UnsafeSubgameState::IsTerminal() const {
  return phase_ == Phase::kSubgame && state_->IsTerminal();
}

std::vector<double> UnsafeSubgameState::Returns() const {
  if (!IsTerminal()) {
    return std::vector<double>(game_->NumPlayers(), 0.0);
  }
  // No scaling in unsafe version - just return subgame utilities directly
  return state_->Returns();
}

std::string UnsafeSubgameState::InformationStateString(Player player) const {
  if (phase_ == Phase::kChance) {
    return "unsafe_start";
  }
  return absl::StrCat("unsafe:subgame:", state_->InformationStateString(player));
}

std::string UnsafeSubgameState::ObservationString(Player player) const {
  return InformationStateString(player);
}

std::string UnsafeSubgameState::ToString() const {
  if (phase_ == Phase::kChance) {
    return "UnsafeSubgame(Chance)";
  }
  return absl::StrCat("UnsafeSubgame(root=", selected_root_, ", ",
                      state_->ToString(), ")");
}

std::unique_ptr<State> UnsafeSubgameState::Clone() const {
  return std::make_unique<UnsafeSubgameState>(*this);
}

std::vector<std::pair<Action, double>> UnsafeSubgameState::ChanceOutcomes()
    const {
  if (phase_ == Phase::kChance) {
    const UnsafeSubgameGame* g = GetUnsafeSubgameGame();
    std::vector<std::pair<Action, double>> outcomes;
    for (int i = 0; i < g->NumSubgameRoots(); ++i) {
      double prob = g->GetRootReachProbability(i) / g->NormalizationConstant();
      outcomes.push_back({i, prob});
    }
    return outcomes;
  } else if (state_->IsChanceNode()) {
    return state_->ChanceOutcomes();
  }
  return {};
}

void UnsafeSubgameState::DoApplyAction(Action action) {
  if (phase_ == Phase::kChance) {
    const UnsafeSubgameGame* g = GetUnsafeSubgameGame();
    selected_root_ = action;
    state_ = g->CloneRootState(action);
    phase_ = Phase::kSubgame;
  } else {
    state_->ApplyAction(action);
  }
}

// =============================================================================
// Factory function
// =============================================================================

std::shared_ptr<const UnsafeSubgameGame> CreateUnsafeSubgame(
    std::shared_ptr<const Game> game,
    std::vector<std::unique_ptr<State>> roots,
    std::vector<double> reach_probs) {
  return std::make_shared<UnsafeSubgameGame>(
      game, std::move(roots), std::move(reach_probs));
}

}  // namespace open_spiel
