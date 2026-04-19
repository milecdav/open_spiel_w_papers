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

#include "open_spiel/game_transforms/resolving_by_is.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/spiel_globals.h"

namespace open_spiel {

namespace {

const GameType kRIBISGameType{
    /*short_name=*/"resolving_by_is",
    /*long_name=*/"Resolving By IS Gadget Game",
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
  GameType new_type = kRIBISGameType;
  new_type.long_name = "ResolvingByIS " + type.long_name;
  new_type.utility = GameType::Utility::kZeroSum;
  new_type.max_num_players = 2;
  new_type.min_num_players = 2;
  new_type.provides_information_state_string = type.provides_information_state_string;
  new_type.provides_observation_string = type.provides_observation_string;
  return new_type;
}

}  // namespace

ResolvingByISGame::ResolvingByISGame(
    std::shared_ptr<const Game> game, std::vector<SubgameRoot> subgame_roots,
    Player resolving_player, std::unordered_map<std::string, double> cfvs)
    : WrappedGame(game, ConvertType(game->GetType()), game->GetParameters()),
      subgame_roots_(std::move(subgame_roots)),
      resolving_player_(resolving_player),
      cfvs_(std::move(cfvs)),
      max_abs_shift_(0.0) {
  SPIEL_CHECK_GT(subgame_roots_.size(), 0);
  SPIEL_CHECK_EQ(game->NumPlayers(), 2);
  SPIEL_CHECK_GE(resolving_player_, 0);
  SPIEL_CHECK_LT(resolving_player_, 2);
  ComputeInfoSetData();
}

void ResolvingByISGame::ComputeInfoSetData() {
  for (int i = 0; i < static_cast<int>(subgame_roots_.size()); ++i) {
    const std::string& is = subgame_roots_[i].info_state_string;
    if (info_set_index_.find(is) == info_set_index_.end()) {
      int idx = info_set_list_.size();
      info_set_list_.push_back(is);
      info_set_index_[is] = idx;
      info_set_roots_.push_back({});
    }
    info_set_roots_[info_set_index_[is]].push_back(i);
    info_set_reach_sums_[is] += subgame_roots_[i].reach_prob;
  }
  for (const auto& is : info_set_list_) {
    double reach_sum = info_set_reach_sums_[is];
    SPIEL_CHECK_GT(reach_sum, 0.0);
    double cfv = 0.0;
    auto it = cfvs_.find(is);
    if (it != cfvs_.end()) cfv = it->second;
    double shift = cfv / reach_sum;
    value_shifts_[is] = shift;
    max_abs_shift_ = std::max(max_abs_shift_, std::abs(shift));
  }
}

std::unique_ptr<State> ResolvingByISGame::NewInitialState() const {
  return std::make_unique<ResolvingByISState>(shared_from_this());
}

int ResolvingByISGame::NumDistinctActions() const {
  return std::max({2, static_cast<int>(info_set_list_.size()),
                   game_->NumDistinctActions()});
}

int ResolvingByISGame::MaxGameLength() const { return 3 + game_->MaxGameLength(); }

int ResolvingByISGame::MaxChanceOutcomes() const {
  int max_roots = 0;
  for (const auto& roots : info_set_roots_) {
    max_roots = std::max(max_roots, static_cast<int>(roots.size()));
  }
  return std::max({static_cast<int>(info_set_list_.size()), max_roots,
                   game_->MaxChanceOutcomes()});
}

double ResolvingByISGame::MinUtility() const {
  return game_->MinUtility() - max_abs_shift_;
}

double ResolvingByISGame::MaxUtility() const {
  return game_->MaxUtility() + max_abs_shift_;
}

const std::string& ResolvingByISGame::InfoSetForAction(int action_idx) const {
  SPIEL_CHECK_GE(action_idx, 0);
  SPIEL_CHECK_LT(action_idx, static_cast<int>(info_set_list_.size()));
  return info_set_list_[action_idx];
}

const std::vector<int>& ResolvingByISGame::RootsForInfoSet(int info_set_idx) const {
  SPIEL_CHECK_GE(info_set_idx, 0);
  SPIEL_CHECK_LT(info_set_idx, static_cast<int>(info_set_roots_.size()));
  return info_set_roots_[info_set_idx];
}

std::unique_ptr<State> ResolvingByISGame::CloneRootState(int root_idx) const {
  SPIEL_CHECK_GE(root_idx, 0);
  SPIEL_CHECK_LT(root_idx, static_cast<int>(subgame_roots_.size()));
  return subgame_roots_[root_idx].state->Clone();
}

double ResolvingByISGame::GetRootReachProbability(int root_idx) const {
  SPIEL_CHECK_GE(root_idx, 0);
  SPIEL_CHECK_LT(root_idx, static_cast<int>(subgame_roots_.size()));
  return subgame_roots_[root_idx].reach_prob;
}

double ResolvingByISGame::GetInfoSetReachSum(const std::string& info_state) const {
  auto it = info_set_reach_sums_.find(info_state);
  if (it != info_set_reach_sums_.end()) return it->second;
  SpielFatalError("Reach sum not found for info state: " + info_state);
}

double ResolvingByISGame::GetValueShift(const std::string& info_state) const {
  auto it = value_shifts_.find(info_state);
  if (it != value_shifts_.end()) return it->second;
  SpielFatalError("Value shift not found for info state: " + info_state);
}

ResolvingByISState::ResolvingByISState(std::shared_ptr<const Game> game)
    : WrappedState(game, nullptr),
      phase_(Phase::kChanceInfoSet),
      chosen_info_set_idx_(-1),
      selected_root_idx_(-1),
      chosen_info_state_(""),
      chose_out_(false) {}

ResolvingByISState::ResolvingByISState(const ResolvingByISState& other)
    : WrappedState(other.game_, other.state_ ? other.state_->Clone() : nullptr),
      phase_(other.phase_),
      chosen_info_set_idx_(other.chosen_info_set_idx_),
      selected_root_idx_(other.selected_root_idx_),
      chosen_info_state_(other.chosen_info_state_),
      chose_out_(other.chose_out_) {
  history_ = other.history_;
}

const ResolvingByISGame* ResolvingByISState::GetRIBISGame() const {
  return down_cast<const ResolvingByISGame*>(game_.get());
}

Player ResolvingByISState::CurrentPlayer() const {
  switch (phase_) {
    case Phase::kChanceInfoSet:
    case Phase::kChanceWithinInfoSet:
      return kChancePlayerId;
    case Phase::kOptionChoice:
      return GetRIBISGame()->NonResolvingPlayer();
    case Phase::kSubgame:
      return state_->CurrentPlayer();
    case Phase::kTerminal:
      return kTerminalPlayerId;
  }
  SpielFatalError("Unknown phase in ResolvingByISState::CurrentPlayer");
}

std::vector<Action> ResolvingByISState::LegalActions() const {
  switch (phase_) {
    case Phase::kChanceInfoSet: {
      std::vector<Action> actions;
      for (int i = 0; i < GetRIBISGame()->NumInfoSets(); ++i) actions.push_back(i);
      return actions;
    }
    case Phase::kOptionChoice:
      return {ResolvingByISGame::kOutAction, ResolvingByISGame::kEnterAction};
    case Phase::kChanceWithinInfoSet: {
      std::vector<Action> actions;
      const auto& roots = GetRIBISGame()->RootsForInfoSet(chosen_info_set_idx_);
      for (int i = 0; i < static_cast<int>(roots.size()); ++i) actions.push_back(i);
      return actions;
    }
    case Phase::kSubgame:
      return state_->LegalActions();
    case Phase::kTerminal:
      return {};
  }
  SpielFatalError("Unknown phase in ResolvingByISState::LegalActions");
}

std::vector<Action> ResolvingByISState::LegalActions(Player player) const {
  if (phase_ == Phase::kSubgame) {
    return state_->LegalActions(player);
  }
  if (player == CurrentPlayer()) return LegalActions();
  return {};
}

std::string ResolvingByISState::ActionToString(Player player, Action action_id) const {
  switch (phase_) {
    case Phase::kChanceInfoSet:
      return absl::StrCat("is_", action_id);
    case Phase::kOptionChoice:
      return action_id == ResolvingByISGame::kOutAction ? "out" : "enter";
    case Phase::kChanceWithinInfoSet:
      return absl::StrCat("root_", action_id);
    case Phase::kSubgame:
      return state_->ActionToString(player, action_id);
    case Phase::kTerminal:
      return "";
  }
  SpielFatalError("Unknown phase in ResolvingByISState::ActionToString");
}

bool ResolvingByISState::IsTerminal() const {
  return phase_ == Phase::kTerminal ||
         (phase_ == Phase::kSubgame && state_->IsTerminal());
}

std::vector<double> ResolvingByISState::Returns() const {
  if (!IsTerminal()) return {0.0, 0.0};
  const auto* g = GetRIBISGame();
  const Player non_res = g->NonResolvingPlayer();
  const double shift = g->GetValueShift(chosen_info_state_);
  if (phase_ == Phase::kTerminal) {
    std::vector<double> r(2, 0.0);
    r[non_res] = shift;
    r[1 - non_res] = -shift;
    return r;
  }
  std::vector<double> r = state_->Returns();
  r[non_res] -= shift;
  r[1 - non_res] += shift;
  return r;
}

std::vector<double> ResolvingByISState::Rewards() const {
  return IsTerminal() ? Returns() : std::vector<double>(2, 0.0);
}

std::string ResolvingByISState::InformationStateString(Player player) const {
  const auto* g = GetRIBISGame();
  if (phase_ == Phase::kChanceInfoSet) {
    return player == g->ResolvingPlayer() ? "ribis_start" : "ribis_chance:root";
  }
  if (phase_ == Phase::kOptionChoice) {
    return player == g->ResolvingPlayer() ? "ribis_choice:opponent_choosing"
                                          : chosen_info_state_;
  }
  if (phase_ == Phase::kChanceWithinInfoSet) {
    return player == g->ResolvingPlayer() ? "ribis_start"
                                          : absl::StrCat("ribis_within:", chosen_info_state_);
  }
  if (phase_ == Phase::kSubgame) {
    return absl::StrCat("ribis_F:subgame:", state_->InformationStateString(player));
  }
  return "ribis_terminal";
}

std::string ResolvingByISState::ObservationString(Player player) const {
  return InformationStateString(player);
}

std::string ResolvingByISState::ToString() const {
  switch (phase_) {
    case Phase::kChanceInfoSet:
      return "RIBIS(ChanceInfoSet)";
    case Phase::kOptionChoice:
      return absl::StrCat("RIBIS(OptionChoice:", chosen_info_state_, ")");
    case Phase::kChanceWithinInfoSet:
      return absl::StrCat("RIBIS(ChanceWithin:", chosen_info_state_, ")");
    case Phase::kSubgame:
      return absl::StrCat("RIBIS(Subgame:", state_->ToString(), ")");
    case Phase::kTerminal:
      return absl::StrCat("RIBIS(TerminalOut:", chosen_info_state_, ")");
  }
  SpielFatalError("Unknown phase in ResolvingByISState::ToString");
}

std::unique_ptr<State> ResolvingByISState::Clone() const {
  return std::make_unique<ResolvingByISState>(*this);
}

std::vector<std::pair<Action, double>> ResolvingByISState::ChanceOutcomes() const {
  const auto* g = GetRIBISGame();
  switch (phase_) {
    case Phase::kChanceInfoSet: {
      std::vector<std::pair<Action, double>> out;
      double total = 0.0;
      for (int i = 0; i < g->NumInfoSets(); ++i) {
        total += g->GetInfoSetReachSum(g->InfoSetForAction(i));
      }
      SPIEL_CHECK_GT(total, 0.0);
      for (int i = 0; i < g->NumInfoSets(); ++i) {
        const std::string& is = g->InfoSetForAction(i);
        out.push_back({i, g->GetInfoSetReachSum(is) / total});
      }
      return out;
    }
    case Phase::kChanceWithinInfoSet: {
      std::vector<std::pair<Action, double>> out;
      const auto& roots = g->RootsForInfoSet(chosen_info_set_idx_);
      double total = g->GetInfoSetReachSum(chosen_info_state_);
      SPIEL_CHECK_GT(total, 0.0);
      for (int i = 0; i < static_cast<int>(roots.size()); ++i) {
        int root_idx = roots[i];
        out.push_back({i, g->GetRootReachProbability(root_idx) / total});
      }
      return out;
    }
    case Phase::kSubgame:
      if (state_->IsChanceNode()) return state_->ChanceOutcomes();
      return {};
    default:
      return {};
  }
}

void ResolvingByISState::DoApplyAction(Action action_id) {
  const auto* g = GetRIBISGame();
  switch (phase_) {
    case Phase::kChanceInfoSet:
      chosen_info_set_idx_ = action_id;
      chosen_info_state_ = g->InfoSetForAction(action_id);
      phase_ = Phase::kOptionChoice;
      return;
    case Phase::kOptionChoice:
      if (action_id == ResolvingByISGame::kOutAction) {
        chose_out_ = true;
        phase_ = Phase::kTerminal;
      } else {
        phase_ = Phase::kChanceWithinInfoSet;
      }
      return;
    case Phase::kChanceWithinInfoSet: {
      const auto& roots = g->RootsForInfoSet(chosen_info_set_idx_);
      SPIEL_CHECK_GE(action_id, 0);
      SPIEL_CHECK_LT(action_id, static_cast<int>(roots.size()));
      selected_root_idx_ = roots[action_id];
      state_ = g->CloneRootState(selected_root_idx_);
      phase_ = Phase::kSubgame;
      return;
    }
    case Phase::kSubgame:
      state_->ApplyAction(action_id);
      return;
    case Phase::kTerminal:
      return;
  }
}

std::shared_ptr<const ResolvingByISGame> CreateResolvingByISGame(
    std::shared_ptr<const Game> game, std::vector<SubgameRoot> subgame_roots,
    Player resolving_player, std::unordered_map<std::string, double> cfvs) {
  return std::make_shared<ResolvingByISGame>(
      std::move(game), std::move(subgame_roots), resolving_player, std::move(cfvs));
}

}  // namespace open_spiel
