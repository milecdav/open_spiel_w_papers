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

#include "open_spiel/game_transforms/ox_gadget.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/algorithms/best_response.h"
#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"

namespace open_spiel {

namespace {

const GameType kOXGadgetGameType{
    /*short_name=*/"ox_gadget",
    /*long_name=*/"OX Gadget Game",
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
  GameType new_type = kOXGadgetGameType;
  new_type.long_name = "OX Gadget " + type.long_name;
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
// OXGadgetGame implementation
// =============================================================================

OXGadgetGame::OXGadgetGame(
    std::shared_ptr<const Game> game,
    std::vector<SubgameRoot> subgame_roots,
    Player resolving_player,
    std::unordered_map<std::string, double> cbv_values,
    double beta,
    std::unordered_map<std::string, double> model_info_set_reach)
    : WrappedGame(game, ConvertType(game->GetType()), game->GetParameters()),
      subgame_roots_(std::move(subgame_roots)),
      resolving_player_(resolving_player),
      cbv_values_(std::move(cbv_values)),
      beta_(beta),
      model_info_set_reach_(std::move(model_info_set_reach)),
      max_abs_shift_(0.0) {
  SPIEL_CHECK_GT(subgame_roots_.size(), 0);
  SPIEL_CHECK_GE(resolving_player_, 0);
  SPIEL_CHECK_LT(resolving_player_, 2);
  SPIEL_CHECK_EQ(game->NumPlayers(), 2);
  SPIEL_CHECK_GT(beta_, 0.0);

  ComputeInfoSetData();
}

void OXGadgetGame::ComputeInfoSetData() {
  // Build ordered list of distinct info sets and index mappings.
  // info_state_string in SubgameRoot is the NON-resolving player's info state.
  for (int i = 0; i < static_cast<int>(subgame_roots_.size()); ++i) {
    const std::string& is = subgame_roots_[i].info_state_string;
    if (info_set_index_.find(is) == info_set_index_.end()) {
      int idx = info_set_list_.size();
      info_set_list_.push_back(is);
      info_set_index_[is] = idx;
      info_set_roots_.push_back({});
    }
    info_set_roots_[info_set_index_[is]].push_back(i);
    // reach_prob is the resolving player's reach (π_{-nonres})
    info_set_reach_sums_[is] += subgame_roots_[i].reach_prob;
  }

  // Compute value shifts: CBV(I) / W(I)
  for (const auto& info_state : info_set_list_) {
    double reach_sum = info_set_reach_sums_[info_state];
    SPIEL_CHECK_GT(reach_sum, 0.0);

    double cbv = 0.0;
    auto it = cbv_values_.find(info_state);
    if (it != cbv_values_.end()) {
      cbv = it->second;
    }

    double shift = cbv / reach_sum;
    value_shifts_[info_state] = shift;
    max_abs_shift_ = std::max(max_abs_shift_, std::abs(shift));
  }

  // Compute exploit chance outcomes: normalize model_info_set_reach over
  // the info sets we actually have.
  double total_model_reach = 0.0;
  for (const auto& info_state : info_set_list_) {
    auto it = model_info_set_reach_.find(info_state);
    if (it != model_info_set_reach_.end()) {
      total_model_reach += it->second;
    }
  }

  // Build normalized exploit outcomes. If total is 0, use uniform.
  if (total_model_reach <= 0.0) {
    double prob = 1.0 / info_set_list_.size();
    for (int i = 0; i < static_cast<int>(info_set_list_.size()); ++i) {
      exploit_chance_outcomes_.push_back({i, prob});
    }
  } else {
    for (int i = 0; i < static_cast<int>(info_set_list_.size()); ++i) {
      const auto& info_state = info_set_list_[i];
      auto it = model_info_set_reach_.find(info_state);
      double reach = (it != model_info_set_reach_.end()) ? it->second : 0.0;
      exploit_chance_outcomes_.push_back({i, reach / total_model_reach});
    }
  }
}

std::unique_ptr<State> OXGadgetGame::NewInitialState() const {
  return std::make_unique<OXGadgetState>(shared_from_this());
}

int OXGadgetGame::NumDistinctActions() const {
  // Max of: 2 (enter/out in option choice), info set choice actions,
  // and original game actions
  return std::max({2, static_cast<int>(info_set_list_.size()),
                   game_->NumDistinctActions()});
}

int OXGadgetGame::MaxGameLength() const {
  // Initial chance (1) + info set chance or chance (1) + option choice (1)
  // + chance within IS (1) + original subgame length
  return 4 + game_->MaxGameLength();
}

int OXGadgetGame::MaxChanceOutcomes() const {
  // Max of: 2 (initial branch), num_info_sets (exploit/safe chance),
  // max roots in any single info set, or original game's chance outcomes
  int max_roots = 0;
  for (const auto& roots : info_set_roots_) {
    max_roots = std::max(max_roots, static_cast<int>(roots.size()));
  }
  return std::max({2, static_cast<int>(info_set_list_.size()),
                   max_roots, game_->MaxChanceOutcomes()});
}

double OXGadgetGame::ExploitBranchProb() const {
  int k = info_set_list_.size();
  return 1.0 / (k * beta_ + 1.0);
}

double OXGadgetGame::SafetyBranchProb() const {
  return 1.0 - ExploitBranchProb();
}

double OXGadgetGame::GetValueShift(const std::string& info_state) const {
  auto it = value_shifts_.find(info_state);
  if (it != value_shifts_.end()) {
    return it->second;
  }
  SpielFatalError("Value shift not found for info state: " + info_state);
}

const std::string& OXGadgetGame::InfoSetForAction(int action_idx) const {
  SPIEL_CHECK_GE(action_idx, 0);
  SPIEL_CHECK_LT(action_idx, static_cast<int>(info_set_list_.size()));
  return info_set_list_[action_idx];
}

const std::vector<int>& OXGadgetGame::RootsForInfoSet(
    int info_set_idx) const {
  SPIEL_CHECK_GE(info_set_idx, 0);
  SPIEL_CHECK_LT(info_set_idx, static_cast<int>(info_set_roots_.size()));
  return info_set_roots_[info_set_idx];
}

std::unique_ptr<State> OXGadgetGame::CloneRootState(int root_idx) const {
  SPIEL_CHECK_GE(root_idx, 0);
  SPIEL_CHECK_LT(root_idx, static_cast<int>(subgame_roots_.size()));
  return subgame_roots_[root_idx].state->Clone();
}

double OXGadgetGame::GetRootReachProbability(int root_idx) const {
  SPIEL_CHECK_GE(root_idx, 0);
  SPIEL_CHECK_LT(root_idx, static_cast<int>(subgame_roots_.size()));
  return subgame_roots_[root_idx].reach_prob;
}

double OXGadgetGame::GetInfoSetReachSum(
    const std::string& info_state) const {
  auto it = info_set_reach_sums_.find(info_state);
  if (it != info_set_reach_sums_.end()) {
    return it->second;
  }
  SpielFatalError("Reach sum not found for info state: " + info_state);
}

double OXGadgetGame::MinUtility() const {
  return game_->MinUtility() - max_abs_shift_;
}

double OXGadgetGame::MaxUtility() const {
  return game_->MaxUtility() + max_abs_shift_;
}

// =============================================================================
// OXGadgetState implementation
// =============================================================================

OXGadgetState::OXGadgetState(std::shared_ptr<const Game> game)
    : WrappedState(game, nullptr),
      phase_(Phase::kInitialChance),
      chosen_branch_(-1),
      chosen_info_set_idx_(-1),
      selected_root_idx_(-1),
      chosen_info_state_(""),
      chose_out_(false) {}

OXGadgetState::OXGadgetState(const OXGadgetState& other)
    : WrappedState(other.game_, other.state_ ? other.state_->Clone() : nullptr),
      phase_(other.phase_),
      chosen_branch_(other.chosen_branch_),
      chosen_info_set_idx_(other.chosen_info_set_idx_),
      selected_root_idx_(other.selected_root_idx_),
      chosen_info_state_(other.chosen_info_state_),
      chose_out_(other.chose_out_) {
  history_ = other.history_;
}

const OXGadgetGame* OXGadgetState::GetOXGame() const {
  return down_cast<const OXGadgetGame*>(game_.get());
}

Player OXGadgetState::CurrentPlayer() const {
  switch (phase_) {
    case Phase::kInitialChance:
      return kChancePlayerId;
    case Phase::kExploitChanceInfoSet:
      return kChancePlayerId;
    case Phase::kSafeChanceInfoSet:
      return kChancePlayerId;
    case Phase::kOptionChoice:
      return GetOXGame()->NonResolvingPlayer();
    case Phase::kChanceWithinInfoSet:
      return kChancePlayerId;
    case Phase::kSubgame:
      return state_->CurrentPlayer();
    case Phase::kTerminal:
      return kTerminalPlayerId;
  }
  SpielFatalError("Unknown phase in OXGadgetState::CurrentPlayer");
}

std::vector<Action> OXGadgetState::LegalActions() const {
  switch (phase_) {
    case Phase::kInitialChance:
      // Two branches: 0=safety, 1=exploit
      return {0, 1};

    case Phase::kExploitChanceInfoSet: {
      // Chance picks info set according to model reach
      const auto& outcomes = GetOXGame()->ExploitChanceOutcomes();
      std::vector<Action> actions;
      actions.reserve(outcomes.size());
      for (const auto& [a, p] : outcomes) {
        actions.push_back(a);
      }
      return actions;
    }

    case Phase::kSafeChanceInfoSet: {
      // Chance picks info set uniformly
      std::vector<Action> actions;
      for (int i = 0; i < GetOXGame()->NumInfoSets(); ++i) {
        actions.push_back(i);
      }
      return actions;
    }

    case Phase::kOptionChoice:
      // Non-resolving player: out (0) or enter (1)
      return {OXGadgetGame::kOutAction, OXGadgetGame::kEnterAction};

    case Phase::kChanceWithinInfoSet: {
      // Chance picks state within the chosen info set
      const auto& roots =
          GetOXGame()->RootsForInfoSet(chosen_info_set_idx_);
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
  SpielFatalError("Unknown phase in OXGadgetState::LegalActions");
}

std::vector<Action> OXGadgetState::LegalActions(Player player) const {
  if (player == CurrentPlayer()) {
    return LegalActions();
  }
  return {};
}

std::string OXGadgetState::ActionToString(Player player,
                                           Action action_id) const {
  switch (phase_) {
    case Phase::kInitialChance:
      return action_id == 0 ? "safety_branch" : "exploit_branch";
    case Phase::kExploitChanceInfoSet:
      return absl::StrCat("exploit_info_set_", action_id);
    case Phase::kSafeChanceInfoSet:
      return absl::StrCat("safe_info_set_", action_id);
    case Phase::kOptionChoice:
      return action_id == OXGadgetGame::kOutAction ? "out" : "enter";
    case Phase::kChanceWithinInfoSet:
      return absl::StrCat("root_", action_id);
    case Phase::kSubgame:
      return state_->ActionToString(player, action_id);
    case Phase::kTerminal:
      return "";
  }
  SpielFatalError("Unknown phase in OXGadgetState::ActionToString");
}

bool OXGadgetState::IsTerminal() const {
  return phase_ == Phase::kTerminal;
}

std::vector<double> OXGadgetState::Returns() const {
  const auto* ox_game = GetOXGame();
  std::vector<double> returns(2, 0.0);

  if (phase_ != Phase::kTerminal) {
    return returns;
  }

  // If non-resolving player chose "out", return {0, 0}
  // (In the shifted game, "out" corresponds to taking CBV - CBV = 0)
  if (chose_out_) {
    return returns;  // already {0, 0}
  }

  // Get subgame returns and apply the shift
  auto subgame_returns = state_->Returns();

  // The shift is CBV(I) / W(I)
  // Non-resolving return -= shift
  // Resolving return += shift (preserving zero-sum)
  double shift = ox_game->GetValueShift(chosen_info_state_);

  returns[ox_game->NonResolvingPlayer()] =
      subgame_returns[ox_game->NonResolvingPlayer()] - shift;
  returns[ox_game->ResolvingPlayer()] =
      subgame_returns[ox_game->ResolvingPlayer()] + shift;

  return returns;
}

std::vector<double> OXGadgetState::Rewards() const {
  if (IsTerminal()) {
    return Returns();
  }
  return std::vector<double>(2, 0.0);
}

std::string OXGadgetState::InformationStateString(Player player) const {
  const auto* ox_game = GetOXGame();
  Player resolving = ox_game->ResolvingPlayer();
  Player non_resolving = ox_game->NonResolvingPlayer();

  switch (phase_) {
    case Phase::kInitialChance:
      if (player == resolving) {
        return "ox_start";
      } else {
        return "ox_chance:top";
      }

    case Phase::kExploitChanceInfoSet:
      if (player == resolving) {
        return "ox_start";
      } else {
        return "ox_exploit:chance";
      }

    case Phase::kSafeChanceInfoSet:
      if (player == resolving) {
        return "ox_start";
      } else {
        return "ox_safe:chance";
      }

    case Phase::kOptionChoice:
      // Non-resolving player makes the enter/out choice
      // Resolving player still sees "ox_start" (cannot distinguish branches)
      if (player == resolving) {
        return "ox_start";
      } else {
        return absl::StrCat("ox_option:", chosen_info_state_);
      }

    case Phase::kChanceWithinInfoSet:
      if (player == resolving) {
        return "ox_start";
      } else {
        if (chosen_branch_ == 1) {
          // Exploit branch
          return absl::StrCat("ox_exploit:within:", chosen_info_state_);
        } else {
          // Safety branch
          return absl::StrCat("ox_safe:within:", chosen_info_state_);
        }
      }

    case Phase::kSubgame:
      // BOTH branches share the same subgame info state prefix
      // This is the crucial OX property: resolving player can't distinguish
      // branches, and non-resolving player is forced to play same strategy
      return absl::StrCat("ox_F:subgame:",
                          state_->InformationStateString(player));

    case Phase::kTerminal:
      if (chose_out_) {
        // "out" terminal has no subgame state
        if (player == resolving) {
          return "ox_start";
        } else {
          return absl::StrCat("ox_option:", chosen_info_state_);
        }
      }
      return absl::StrCat("ox_F:subgame:",
                          state_->InformationStateString(player));
  }
  SpielFatalError("Unknown phase in OXGadgetState::InformationStateString");
}

std::string OXGadgetState::ObservationString(Player player) const {
  return InformationStateString(player);
}

std::string OXGadgetState::ToString() const {
  std::string result = "OXGadgetState(";
  switch (phase_) {
    case Phase::kInitialChance:
      result += "phase=InitialChance";
      break;
    case Phase::kExploitChanceInfoSet:
      result += "phase=ExploitChanceInfoSet";
      break;
    case Phase::kSafeChanceInfoSet:
      result += "phase=SafeChanceInfoSet";
      break;
    case Phase::kOptionChoice:
      result += absl::StrCat("phase=OptionChoice, info_set=",
                              chosen_info_state_);
      break;
    case Phase::kChanceWithinInfoSet:
      result += absl::StrCat("phase=ChanceWithinInfoSet, branch=",
                              chosen_branch_, ", info_set=",
                              chosen_info_set_idx_,
                              ", info=", chosen_info_state_);
      break;
    case Phase::kSubgame:
      result += absl::StrCat("phase=Subgame, branch=", chosen_branch_,
                              ", root=", selected_root_idx_,
                              ", state=", state_->ToString());
      break;
    case Phase::kTerminal:
      result += absl::StrCat("phase=Terminal, branch=", chosen_branch_,
                              ", root=", selected_root_idx_,
                              ", chose_out=", chose_out_);
      break;
  }
  result += ")";
  return result;
}

std::unique_ptr<State> OXGadgetState::Clone() const {
  return std::make_unique<OXGadgetState>(*this);
}

std::vector<std::pair<Action, double>> OXGadgetState::ChanceOutcomes() const {
  const auto* ox_game = GetOXGame();

  switch (phase_) {
    case Phase::kInitialChance: {
      double exploit_prob = ox_game->ExploitBranchProb();
      double safety_prob = ox_game->SafetyBranchProb();
      // Action 0 = safety branch, Action 1 = exploit branch
      return {{0, safety_prob}, {1, exploit_prob}};
    }

    case Phase::kExploitChanceInfoSet:
      return ox_game->ExploitChanceOutcomes();

    case Phase::kSafeChanceInfoSet: {
      // Chance picks info set uniformly
      std::vector<std::pair<Action, double>> outcomes;
      double prob = 1.0 / ox_game->NumInfoSets();
      for (int i = 0; i < ox_game->NumInfoSets(); ++i) {
        outcomes.push_back({i, prob});
      }
      return outcomes;
    }

    case Phase::kChanceWithinInfoSet: {
      const auto& roots = ox_game->RootsForInfoSet(chosen_info_set_idx_);
      double reach_sum = ox_game->GetInfoSetReachSum(chosen_info_state_);

      std::vector<std::pair<Action, double>> outcomes;
      for (int i = 0; i < static_cast<int>(roots.size()); ++i) {
        double prob = ox_game->GetRootReachProbability(roots[i]) / reach_sum;
        outcomes.push_back({i, prob});
      }
      return outcomes;
    }

    case Phase::kSubgame:
      if (state_->IsChanceNode()) {
        return state_->ChanceOutcomes();
      }
      return {};

    default:
      return {};
  }
}

void OXGadgetState::DoApplyAction(Action action_id) {
  const auto* ox_game = GetOXGame();

  switch (phase_) {
    case Phase::kInitialChance:
      // Chance picks branch: 0=safety, 1=exploit
      chosen_branch_ = action_id;
      if (chosen_branch_ == 0) {
        // Safety branch: chance picks IS uniformly
        phase_ = Phase::kSafeChanceInfoSet;
      } else {
        // Exploit branch: chance picks IS by model reach
        phase_ = Phase::kExploitChanceInfoSet;
      }
      break;

    case Phase::kExploitChanceInfoSet:
      // Chance selects info set weighted by model reach
      chosen_info_set_idx_ = action_id;
      chosen_info_state_ = ox_game->InfoSetForAction(action_id);
      phase_ = Phase::kChanceWithinInfoSet;
      break;

    case Phase::kSafeChanceInfoSet:
      // Chance selects info set uniformly
      chosen_info_set_idx_ = action_id;
      chosen_info_state_ = ox_game->InfoSetForAction(action_id);
      phase_ = Phase::kOptionChoice;
      break;

    case Phase::kOptionChoice:
      if (action_id == OXGadgetGame::kOutAction) {
        chose_out_ = true;
        phase_ = Phase::kTerminal;
      } else {
        // enter: proceed to chance within info set
        phase_ = Phase::kChanceWithinInfoSet;
      }
      break;

    case Phase::kChanceWithinInfoSet: {
      // Chance selects which root state within the chosen info set
      const auto& roots = ox_game->RootsForInfoSet(chosen_info_set_idx_);
      selected_root_idx_ = roots[action_id];
      state_ = ox_game->CloneRootState(selected_root_idx_);
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

std::shared_ptr<const OXGadgetGame> CreateOXGadgetGame(
    std::shared_ptr<const Game> game,
    std::vector<SubgameRoot> subgame_roots,
    Player resolving_player,
    std::unordered_map<std::string, double> cbv_values,
    double beta,
    std::unordered_map<std::string, double> model_info_set_reach) {
  return std::make_shared<OXGadgetGame>(
      game, std::move(subgame_roots), resolving_player,
      std::move(cbv_values), beta, std::move(model_info_set_reach));
}

// =============================================================================
// ResolveWithOXGadget
// =============================================================================

std::shared_ptr<TabularPolicy> ResolveWithOXGadget(
    const SubgameDecomposition& decomp,
    const TabularPolicy& trunk_policy,
    double beta,
    const Policy& opponent_model,
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
  for (int non_res = 0; non_res < 2; ++non_res) {
    int res = 1 - non_res;

    // Compute CBV for the non-resolving player using TabularBestResponse
    // BR is computed on the FULL game against the trunk policy (blueprint)
    algorithms::TabularBestResponse br(*decomp.game, non_res, &trunk_policy);

    for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
      auto info_per_player = CollectSubgameInfoStatesPerPlayer(roots);

      // Build SubgameRoots for OX:
      // - info_state_string = NON-resolving player's info state (for grouping)
      // - reach_prob = resolving player's reach (π_{-nonres})
      auto ox_roots =
          BuildSubgameRoots(roots, non_res, decomp.reach_probs[res]);
      if (ox_roots.empty()) continue;

      // Compute CBV: CBV(I) = Σ_{h∈I} π_{-nonres}(h) × br.Value(h)
      std::unordered_map<std::string, double> cbv_values;
      for (const auto& root : roots) {
        std::string info_state = root->InformationStateString(non_res);
        std::string hist = root->HistoryString();
        auto reach_it = decomp.reach_probs[res].find(hist);
        double reach =
            (reach_it != decomp.reach_probs[res].end()) ? reach_it->second
                                                         : 0.0;
        if (reach > 0) {
          cbv_values[info_state] += reach * br.Value(hist);
        }
      }

      // Compute model info set reach
      std::vector<const State*> root_ptrs;
      for (const auto& root : roots) root_ptrs.push_back(root.get());
      auto model_reach = ComputeReachProbabilities(
          *decomp.game, opponent_model, non_res, root_ptrs);

      std::unordered_map<std::string, double> model_info_set_reach;
      for (const auto& root : roots) {
        std::string non_res_is = root->InformationStateString(non_res);
        auto it = model_reach.find(root->HistoryString());
        if (it != model_reach.end()) {
          model_info_set_reach[non_res_is] += it->second;
        }
      }

      auto ox_game = CreateOXGadgetGame(
          decomp.game, std::move(ox_roots), res, cbv_values,
          beta, model_info_set_reach);

      algorithms::CFRSolverBase solver(*ox_game, true, true, true);
      for (int i = 0; i < cfr_iterations; ++i) {
        solver.EvaluateAndUpdatePolicy();
      }

      // Extract RESOLVING player's strategy
      const std::string prefix = "ox_F:subgame:";
      TabularPolicy policy = solver.TabularAveragePolicy();
      for (const auto& [sub_is, ap] : policy.PolicyTable()) {
        if (sub_is.compare(0, prefix.length(), prefix) == 0) {
          std::string orig = sub_is.substr(prefix.length());
          if (info_per_player[res].count(orig) > 0) {
            combined->SetStatePolicy(orig, ap);
          }
        }
      }
    }
  }

  return combined;
}

}  // namespace open_spiel
