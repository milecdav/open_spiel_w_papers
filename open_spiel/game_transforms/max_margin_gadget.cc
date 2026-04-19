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

#include "open_spiel/game_transforms/max_margin_gadget.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"

namespace open_spiel {

namespace {

const GameType kMaxMarginGadgetGameType{
    /*short_name=*/"max_margin_gadget",
    /*long_name=*/"Max-Margin Gadget Game",
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
  GameType new_type = kMaxMarginGadgetGameType;
  new_type.long_name = "Max-Margin Gadget " + type.long_name;
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
// MaxMarginGadgetGame implementation
// =============================================================================

MaxMarginGadgetGame::MaxMarginGadgetGame(
    std::shared_ptr<const Game> game,
    std::vector<SubgameRoot> subgame_roots,
    Player adversary_player,
    std::unordered_map<std::string, double> counterfactual_values)
    : WrappedGame(game, ConvertType(game->GetType()), game->GetParameters()),
      subgame_roots_(std::move(subgame_roots)),
      adversary_player_(adversary_player),
      counterfactual_values_(std::move(counterfactual_values)),
      max_abs_shift_(0.0) {
  SPIEL_CHECK_GT(subgame_roots_.size(), 0);
  SPIEL_CHECK_GE(adversary_player_, 0);
  SPIEL_CHECK_LT(adversary_player_, 2);
  SPIEL_CHECK_EQ(game->NumPlayers(), 2);

  ComputeInfoSetData();
}

void MaxMarginGadgetGame::ComputeInfoSetData() {
  // Build ordered list of distinct info sets and index mappings
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

  // Compute value shifts: CFV(I) / W(I)
  for (const auto& info_state : info_set_list_) {
    double reach_sum = info_set_reach_sums_[info_state];
    SPIEL_CHECK_GT(reach_sum, 0.0);

    double cf_value = 0.0;
    auto it = counterfactual_values_.find(info_state);
    if (it != counterfactual_values_.end()) {
      cf_value = it->second;
    }

    double shift = cf_value / reach_sum;
    value_shifts_[info_state] = shift;
    max_abs_shift_ = std::max(max_abs_shift_, std::abs(shift));
  }
}

std::unique_ptr<State> MaxMarginGadgetGame::NewInitialState() const {
  return std::make_unique<MaxMarginGadgetState>(shared_from_this());
}

int MaxMarginGadgetGame::NumDistinctActions() const {
  // Max of info set choice actions and original game actions
  return std::max(static_cast<int>(info_set_list_.size()),
                  game_->NumDistinctActions());
}

int MaxMarginGadgetGame::MaxGameLength() const {
  // Info set choice (1) + chance (1) + original subgame length
  return 2 + game_->MaxGameLength();
}

int MaxMarginGadgetGame::MaxChanceOutcomes() const {
  // Max roots in any single info set, or original game's chance outcomes
  int max_roots = 0;
  for (const auto& roots : info_set_roots_) {
    max_roots = std::max(max_roots, static_cast<int>(roots.size()));
  }
  return std::max(max_roots, game_->MaxChanceOutcomes());
}

double MaxMarginGadgetGame::GetValueShift(
    const std::string& info_state) const {
  auto it = value_shifts_.find(info_state);
  if (it != value_shifts_.end()) {
    return it->second;
  }
  SpielFatalError("Value shift not found for info state: " + info_state);
}

const std::string& MaxMarginGadgetGame::InfoSetForAction(
    int action_idx) const {
  SPIEL_CHECK_GE(action_idx, 0);
  SPIEL_CHECK_LT(action_idx, static_cast<int>(info_set_list_.size()));
  return info_set_list_[action_idx];
}

const std::vector<int>& MaxMarginGadgetGame::RootsForInfoSet(
    int info_set_idx) const {
  SPIEL_CHECK_GE(info_set_idx, 0);
  SPIEL_CHECK_LT(info_set_idx, static_cast<int>(info_set_roots_.size()));
  return info_set_roots_[info_set_idx];
}

std::unique_ptr<State> MaxMarginGadgetGame::CloneRootState(
    int root_idx) const {
  SPIEL_CHECK_GE(root_idx, 0);
  SPIEL_CHECK_LT(root_idx, static_cast<int>(subgame_roots_.size()));
  return subgame_roots_[root_idx].state->Clone();
}

double MaxMarginGadgetGame::GetRootReachProbability(int root_idx) const {
  SPIEL_CHECK_GE(root_idx, 0);
  SPIEL_CHECK_LT(root_idx, static_cast<int>(subgame_roots_.size()));
  return subgame_roots_[root_idx].reach_prob;
}

double MaxMarginGadgetGame::GetInfoSetReachSum(
    const std::string& info_state) const {
  auto it = info_set_reach_sums_.find(info_state);
  if (it != info_set_reach_sums_.end()) {
    return it->second;
  }
  SpielFatalError("Reach sum not found for info state: " + info_state);
}

double MaxMarginGadgetGame::MinUtility() const {
  return game_->MinUtility() - max_abs_shift_;
}

double MaxMarginGadgetGame::MaxUtility() const {
  return game_->MaxUtility() + max_abs_shift_;
}

// =============================================================================
// MaxMarginGadgetState implementation
// =============================================================================

MaxMarginGadgetState::MaxMarginGadgetState(std::shared_ptr<const Game> game)
    : WrappedState(game, nullptr),
      phase_(Phase::kInfoSetChoice),
      chosen_info_set_idx_(-1),
      selected_root_idx_(-1),
      chosen_info_state_("") {}

MaxMarginGadgetState::MaxMarginGadgetState(const MaxMarginGadgetState& other)
    : WrappedState(other.game_, other.state_ ? other.state_->Clone() : nullptr),
      phase_(other.phase_),
      chosen_info_set_idx_(other.chosen_info_set_idx_),
      selected_root_idx_(other.selected_root_idx_),
      chosen_info_state_(other.chosen_info_state_) {
  history_ = other.history_;
}

const MaxMarginGadgetGame* MaxMarginGadgetState::GetMaxMarginGame() const {
  return down_cast<const MaxMarginGadgetGame*>(game_.get());
}

Player MaxMarginGadgetState::CurrentPlayer() const {
  switch (phase_) {
    case Phase::kInfoSetChoice:
      // The adversary (non-resolving player) picks info sets
      return GetMaxMarginGame()->NonResolvingPlayer();
    case Phase::kChance:
      return kChancePlayerId;
    case Phase::kSubgame:
      return state_->CurrentPlayer();
    case Phase::kTerminal:
      return kTerminalPlayerId;
  }
  SpielFatalError("Unknown phase in MaxMarginGadgetState::CurrentPlayer");
}

std::vector<Action> MaxMarginGadgetState::LegalActions() const {
  switch (phase_) {
    case Phase::kInfoSetChoice: {
      // One action per distinct info set
      std::vector<Action> actions;
      for (int i = 0; i < GetMaxMarginGame()->NumInfoSets(); ++i) {
        actions.push_back(i);
      }
      return actions;
    }
    case Phase::kChance: {
      // Actions are local indices into roots for the chosen info set
      const auto& roots =
          GetMaxMarginGame()->RootsForInfoSet(chosen_info_set_idx_);
      std::vector<Action> actions;
      for (int i = 0; i < static_cast<int>(roots.size()); ++i) {
        actions.push_back(i);
      }
      return actions;
    }
    case Phase::kSubgame:
      return state_->LegalActions();
    case Phase::kTerminal:
      return {};
  }
  SpielFatalError("Unknown phase in MaxMarginGadgetState::LegalActions");
}

std::vector<Action> MaxMarginGadgetState::LegalActions(Player player) const {
  if (phase_ == Phase::kSubgame) {
    return state_->LegalActions(player);
  }
  if (player == CurrentPlayer()) {
    return LegalActions();
  }
  return {};
}

std::string MaxMarginGadgetState::ActionToString(Player player,
                                                  Action action_id) const {
  switch (phase_) {
    case Phase::kInfoSetChoice:
      return absl::StrCat("info_set_", action_id);
    case Phase::kChance:
      return absl::StrCat("root_", action_id);
    case Phase::kSubgame:
      return state_->ActionToString(player, action_id);
    case Phase::kTerminal:
      return "";
  }
  SpielFatalError("Unknown phase in MaxMarginGadgetState::ActionToString");
}

bool MaxMarginGadgetState::IsTerminal() const {
  return phase_ == Phase::kTerminal;
}

std::vector<double> MaxMarginGadgetState::Returns() const {
  const auto* mm_game = GetMaxMarginGame();
  std::vector<double> returns(2, 0.0);

  if (phase_ != Phase::kTerminal) {
    return returns;
  }

  // Get subgame returns and apply the shift
  auto subgame_returns = state_->Returns();
  double shift = mm_game->GetValueShift(chosen_info_state_);

  // Non-resolving (adversary) player: subgame return - shift (= original - CFV/W)
  // Resolving player: subgame return + shift (zero-sum preserved)
  returns[mm_game->NonResolvingPlayer()] =
      subgame_returns[mm_game->NonResolvingPlayer()] - shift;
  returns[mm_game->ResolvingPlayer()] =
      subgame_returns[mm_game->ResolvingPlayer()] + shift;

  return returns;
}

std::vector<double> MaxMarginGadgetState::Rewards() const {
  if (IsTerminal()) {
    return Returns();
  }
  return std::vector<double>(2, 0.0);
}

std::string MaxMarginGadgetState::InformationStateString(Player player) const {
  const auto* mm_game = GetMaxMarginGame();

  switch (phase_) {
    case Phase::kInfoSetChoice:
      if (player == mm_game->NonResolvingPlayer()) {
        return "mm_start";
      } else {
        return "mm_choice:opponent_choosing";
      }

    case Phase::kChance:
      if (player == mm_game->NonResolvingPlayer()) {
        return absl::StrCat("mm_choice:", chosen_info_state_);
      } else {
        return "mm_choice:opponent_choosing";
      }

    case Phase::kSubgame:
      return absl::StrCat("mm_F:subgame:",
                          state_->InformationStateString(player));

    case Phase::kTerminal:
      return absl::StrCat("mm_F:subgame:",
                          state_->InformationStateString(player));
  }
  SpielFatalError(
      "Unknown phase in MaxMarginGadgetState::InformationStateString");
}

std::string MaxMarginGadgetState::ObservationString(Player player) const {
  return InformationStateString(player);
}

std::string MaxMarginGadgetState::ToString() const {
  std::string result = "MaxMarginGadgetState(";
  switch (phase_) {
    case Phase::kInfoSetChoice:
      result += "phase=InfoSetChoice";
      break;
    case Phase::kChance:
      result += absl::StrCat("phase=Chance, info_set=", chosen_info_set_idx_,
                              ", info=", chosen_info_state_);
      break;
    case Phase::kSubgame:
      result += absl::StrCat("phase=Subgame, root=", selected_root_idx_,
                              ", state=", state_->ToString());
      break;
    case Phase::kTerminal:
      result += absl::StrCat("phase=Terminal, root=", selected_root_idx_);
      break;
  }
  result += ")";
  return result;
}

std::unique_ptr<State> MaxMarginGadgetState::Clone() const {
  return std::make_unique<MaxMarginGadgetState>(*this);
}

std::vector<std::pair<Action, double>>
MaxMarginGadgetState::ChanceOutcomes() const {
  if (phase_ == Phase::kChance) {
    const auto* mm_game = GetMaxMarginGame();
    const auto& roots = mm_game->RootsForInfoSet(chosen_info_set_idx_);
    double reach_sum = mm_game->GetInfoSetReachSum(chosen_info_state_);

    std::vector<std::pair<Action, double>> outcomes;
    for (int i = 0; i < static_cast<int>(roots.size()); ++i) {
      double prob = mm_game->GetRootReachProbability(roots[i]) / reach_sum;
      outcomes.push_back({i, prob});
    }
    return outcomes;
  } else if (phase_ == Phase::kSubgame && state_->IsChanceNode()) {
    return state_->ChanceOutcomes();
  }
  return {};
}

void MaxMarginGadgetState::DoApplyAction(Action action_id) {
  const auto* mm_game = GetMaxMarginGame();

  switch (phase_) {
    case Phase::kInfoSetChoice:
      // Non-resolving (adversary) player selects which info set to challenge
      chosen_info_set_idx_ = action_id;
      chosen_info_state_ = mm_game->InfoSetForAction(action_id);
      phase_ = Phase::kChance;
      break;

    case Phase::kChance: {
      // Chance selects which root state within the chosen info set
      const auto& roots = mm_game->RootsForInfoSet(chosen_info_set_idx_);
      selected_root_idx_ = roots[action_id];
      state_ = mm_game->CloneRootState(selected_root_idx_);
      if (state_->IsTerminal()) {
        phase_ = Phase::kTerminal;
      } else {
        phase_ = Phase::kSubgame;
      }
      break;
    }

    case Phase::kSubgame:
      state_->ApplyAction(action_id);
      if (state_->IsTerminal()) {
        phase_ = Phase::kTerminal;
      }
      break;

    case Phase::kTerminal:
      SpielFatalError("Cannot apply action to terminal state");
  }
}

// =============================================================================
// Factory function
// =============================================================================

std::shared_ptr<const MaxMarginGadgetGame> CreateMaxMarginGadgetGame(
    std::shared_ptr<const Game> game,
    std::vector<SubgameRoot> subgame_roots,
    Player adversary_player,
    std::unordered_map<std::string, double> counterfactual_values) {
  return std::make_shared<MaxMarginGadgetGame>(
      game, std::move(subgame_roots), adversary_player,
      std::move(counterfactual_values));
}

// =============================================================================
// ResolveWithMaxMarginGadget
// =============================================================================

std::shared_ptr<TabularPolicy> ResolveWithMaxMarginGadget(
    const SubgameDecomposition& decomp,
    const TabularPolicy& trunk_policy,
    int cfr_iterations) {
  auto combined = std::make_shared<TabularPolicy>();

  // Copy trunk
  for (const auto& [player, info_states] : decomp.trunk_info_states) {
    for (const auto& is : info_states) {
      auto ap = trunk_policy.GetStatePolicy(is);
      if (!ap.empty()) combined->SetStatePolicy(is, ap);
    }
  }

  // Re-solve for both players
  for (int res = 0; res < 2; ++res) {
    int non_res = 1 - res;

    for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
      auto info_per_player = CollectSubgameInfoStatesPerPlayer(roots);
      // Roots grouped by non-resolving (adversary) player's info states,
      // with resolving player's reach probabilities
      auto gadget_roots =
          BuildSubgameRoots(roots, non_res, decomp.reach_probs[res]);
      if (gadget_roots.empty()) continue;

      // adversary_player = non_res (picks info sets)
      auto mm_gadget = CreateMaxMarginGadgetGame(
          decomp.game, std::move(gadget_roots), non_res, decomp.cfvs[non_res]);
      algorithms::CFRSolverBase solver(*mm_gadget, true, true, true);
      for (int i = 0; i < cfr_iterations; ++i) {
        solver.EvaluateAndUpdatePolicy();
      }

      // Extract the resolving player's strategy from the gadget
      TabularPolicy gadget_policy = solver.TabularAveragePolicy();
      const std::string prefix = "mm_F:subgame:";
      for (const auto& [gadget_is, ap] : gadget_policy.PolicyTable()) {
        if (gadget_is.find(prefix) == 0) {
          std::string orig_is = gadget_is.substr(prefix.length());
          if (info_per_player[res].count(orig_is) > 0) {
            combined->SetStatePolicy(orig_is, ap);
          }
        }
      }
    }
  }

  return combined;
}

}  // namespace open_spiel
