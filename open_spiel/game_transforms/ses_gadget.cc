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

#include "open_spiel/game_transforms/ses_gadget.h"

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

const GameType kSESGadgetGameType{
    /*short_name=*/"ses_gadget",
    /*long_name=*/"SES Gadget Game",
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
  GameType new_type = kSESGadgetGameType;
  new_type.long_name = "SES Gadget " + type.long_name;
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
// SESGadgetGame implementation
// =============================================================================

SESGadgetGame::SESGadgetGame(
    std::shared_ptr<const Game> game,
    std::vector<SubgameRoot> subgame_roots,
    Player resolving_player,
    std::unordered_map<std::string, double> counterfactual_values,
    double alpha,
    std::unordered_map<std::string, double> model_info_set_reach)
    : WrappedGame(game, ConvertType(game->GetType()), game->GetParameters()),
      subgame_roots_(std::move(subgame_roots)),
      resolving_player_(resolving_player),
      counterfactual_values_(std::move(counterfactual_values)),
      alpha_(alpha),
      model_info_set_reach_(std::move(model_info_set_reach)),
      max_abs_shift_(0.0) {
  SPIEL_CHECK_GT(subgame_roots_.size(), 0);
  SPIEL_CHECK_GE(resolving_player_, 0);
  SPIEL_CHECK_LT(resolving_player_, 2);
  SPIEL_CHECK_EQ(game->NumPlayers(), 2);
  SPIEL_CHECK_GE(alpha_, 0.0);
  SPIEL_CHECK_LE(alpha_, 1.0);

  ComputeInfoSetData();
}

void SESGadgetGame::ComputeInfoSetData() {
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

  // Compute value shifts: CFV_nonres(I) / W(I)
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

  // Compute exploit chance outcomes: normalize model_info_set_reach over
  // the info sets we actually have.
  double total_model_reach = 0.0;
  for (const auto& info_state : info_set_list_) {
    auto it = model_info_set_reach_.find(info_state);
    if (it != model_info_set_reach_.end()) {
      total_model_reach += it->second;
    }
  }

  // Build normalized exploit outcomes. Zero total model reach indicates an
  // invalid/empty model policy projection and should fail fast.
  if (total_model_reach <= 0.0) {
    SpielFatalError("SES gadget: total_model_reach <= 0 while building outcomes");
  } else {
    for (int i = 0; i < static_cast<int>(info_set_list_.size()); ++i) {
      const auto& info_state = info_set_list_[i];
      auto it = model_info_set_reach_.find(info_state);
      double reach = (it != model_info_set_reach_.end()) ? it->second : 0.0;
      exploit_chance_outcomes_.push_back({i, reach / total_model_reach});
    }
  }
}

std::unique_ptr<State> SESGadgetGame::NewInitialState() const {
  return std::make_unique<SESGadgetState>(shared_from_this());
}

int SESGadgetGame::NumDistinctActions() const {
  // Max of info set choice actions (for non-resolving in safety branch)
  // and original game actions
  return std::max(static_cast<int>(info_set_list_.size()),
                  game_->NumDistinctActions());
}

int SESGadgetGame::MaxGameLength() const {
  // Initial chance (1) + info set choice or chance (1) + chance within IS (1)
  // + original subgame length
  return 3 + game_->MaxGameLength();
}

int SESGadgetGame::MaxChanceOutcomes() const {
  // Max of: 2 (initial branch), num_info_sets (exploit chance),
  // max roots in any single info set, or original game's chance outcomes
  int max_roots = 0;
  for (const auto& roots : info_set_roots_) {
    max_roots = std::max(max_roots, static_cast<int>(roots.size()));
  }
  return std::max({2, static_cast<int>(info_set_list_.size()),
                   max_roots, game_->MaxChanceOutcomes()});
}

double SESGadgetGame::GetValueShift(const std::string& info_state) const {
  auto it = value_shifts_.find(info_state);
  if (it != value_shifts_.end()) {
    return it->second;
  }
  SpielFatalError("Value shift not found for info state: " + info_state);
}

const std::string& SESGadgetGame::InfoSetForAction(int action_idx) const {
  SPIEL_CHECK_GE(action_idx, 0);
  SPIEL_CHECK_LT(action_idx, static_cast<int>(info_set_list_.size()));
  return info_set_list_[action_idx];
}

const std::vector<int>& SESGadgetGame::RootsForInfoSet(
    int info_set_idx) const {
  SPIEL_CHECK_GE(info_set_idx, 0);
  SPIEL_CHECK_LT(info_set_idx, static_cast<int>(info_set_roots_.size()));
  return info_set_roots_[info_set_idx];
}

std::unique_ptr<State> SESGadgetGame::CloneRootState(int root_idx) const {
  SPIEL_CHECK_GE(root_idx, 0);
  SPIEL_CHECK_LT(root_idx, static_cast<int>(subgame_roots_.size()));
  return subgame_roots_[root_idx].state->Clone();
}

double SESGadgetGame::GetRootReachProbability(int root_idx) const {
  SPIEL_CHECK_GE(root_idx, 0);
  SPIEL_CHECK_LT(root_idx, static_cast<int>(subgame_roots_.size()));
  return subgame_roots_[root_idx].reach_prob;
}

double SESGadgetGame::GetInfoSetReachSum(
    const std::string& info_state) const {
  auto it = info_set_reach_sums_.find(info_state);
  if (it != info_set_reach_sums_.end()) {
    return it->second;
  }
  SpielFatalError("Reach sum not found for info state: " + info_state);
}

double SESGadgetGame::MinUtility() const {
  return game_->MinUtility() - max_abs_shift_;
}

double SESGadgetGame::MaxUtility() const {
  return game_->MaxUtility() + max_abs_shift_;
}

// =============================================================================
// SESGadgetState implementation
// =============================================================================

SESGadgetState::SESGadgetState(std::shared_ptr<const Game> game)
    : WrappedState(game, nullptr),
      phase_(Phase::kInitialChance),
      chosen_branch_(-1),
      chosen_info_set_idx_(-1),
      selected_root_idx_(-1),
      chosen_info_state_("") {}

SESGadgetState::SESGadgetState(const SESGadgetState& other)
    : WrappedState(other.game_, other.state_ ? other.state_->Clone() : nullptr),
      phase_(other.phase_),
      chosen_branch_(other.chosen_branch_),
      chosen_info_set_idx_(other.chosen_info_set_idx_),
      selected_root_idx_(other.selected_root_idx_),
      chosen_info_state_(other.chosen_info_state_) {
  history_ = other.history_;
}

const SESGadgetGame* SESGadgetState::GetSESGame() const {
  return down_cast<const SESGadgetGame*>(game_.get());
}

Player SESGadgetState::CurrentPlayer() const {
  switch (phase_) {
    case Phase::kInitialChance:
      return kChancePlayerId;
    case Phase::kSafeInfoSetChoice:
      return GetSESGame()->NonResolvingPlayer();
    case Phase::kExploitChanceInfoSet:
      return kChancePlayerId;
    case Phase::kChanceWithinInfoSet:
      return kChancePlayerId;
    case Phase::kSubgame:
      return state_->CurrentPlayer();
    case Phase::kTerminal:
      return kTerminalPlayerId;
  }
  SpielFatalError("Unknown phase in SESGadgetState::CurrentPlayer");
}

std::vector<Action> SESGadgetState::LegalActions() const {
  switch (phase_) {
    case Phase::kInitialChance:
      // Two branches: 0=safety, 1=exploit
      // If alpha == 0, only safety (but we still have chance node with 2
      // actions, one with prob 0 is fine; alternatively just 1 action).
      // For simplicity always return 2 actions.
      return {0, 1};

    case Phase::kSafeInfoSetChoice: {
      // Non-resolving player picks one of the info sets
      std::vector<Action> actions;
      for (int i = 0; i < GetSESGame()->NumInfoSets(); ++i) {
        actions.push_back(i);
      }
      return actions;
    }

    case Phase::kExploitChanceInfoSet: {
      // Chance picks info set according to model reach
      const auto& outcomes = GetSESGame()->ExploitChanceOutcomes();
      std::vector<Action> actions;
      actions.reserve(outcomes.size());
      for (const auto& [a, p] : outcomes) {
        actions.push_back(a);
      }
      return actions;
    }

    case Phase::kChanceWithinInfoSet: {
      // Chance picks state within the chosen info set
      const auto& roots =
          GetSESGame()->RootsForInfoSet(chosen_info_set_idx_);
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
  SpielFatalError("Unknown phase in SESGadgetState::LegalActions");
}

std::vector<Action> SESGadgetState::LegalActions(Player player) const {
  if (player == CurrentPlayer()) {
    return LegalActions();
  }
  return {};
}

std::string SESGadgetState::ActionToString(Player player,
                                            Action action_id) const {
  switch (phase_) {
    case Phase::kInitialChance:
      return action_id == 0 ? "safety_branch" : "exploit_branch";
    case Phase::kSafeInfoSetChoice:
      return absl::StrCat("safe_info_set_", action_id);
    case Phase::kExploitChanceInfoSet:
      return absl::StrCat("exploit_info_set_", action_id);
    case Phase::kChanceWithinInfoSet:
      return absl::StrCat("root_", action_id);
    case Phase::kSubgame:
      return state_->ActionToString(player, action_id);
    case Phase::kTerminal:
      return "";
  }
  SpielFatalError("Unknown phase in SESGadgetState::ActionToString");
}

bool SESGadgetState::IsTerminal() const {
  return phase_ == Phase::kTerminal;
}

std::vector<double> SESGadgetState::Returns() const {
  const auto* ses_game = GetSESGame();
  std::vector<double> returns(2, 0.0);

  if (phase_ != Phase::kTerminal) {
    return returns;
  }

  // Get subgame returns and apply the shift
  auto subgame_returns = state_->Returns();

  // The shift is CFV_nonres(I) / W(I)
  // Non-resolving return -= shift (paper subtracts v^σ_1(I) from u_1(z))
  // Resolving return += shift (preserving zero-sum)
  double shift = ses_game->GetValueShift(chosen_info_state_);

  returns[ses_game->NonResolvingPlayer()] =
      subgame_returns[ses_game->NonResolvingPlayer()] - shift;
  returns[ses_game->ResolvingPlayer()] =
      subgame_returns[ses_game->ResolvingPlayer()] + shift;

  return returns;
}

std::vector<double> SESGadgetState::Rewards() const {
  if (IsTerminal()) {
    return Returns();
  }
  return std::vector<double>(2, 0.0);
}

std::string SESGadgetState::InformationStateString(Player player) const {
  const auto* ses_game = GetSESGame();
  Player resolving = ses_game->ResolvingPlayer();
  Player non_resolving = ses_game->NonResolvingPlayer();

  switch (phase_) {
    case Phase::kInitialChance:
      // This is a chance node; called for information purposes only.
      // The resolving player can't tell which branch they're in.
      // The non-resolving player sees the top-level chance.
      if (player == resolving) {
        return "ses_start";
      } else {
        return "ses_chance:top";
      }

    case Phase::kSafeInfoSetChoice:
      // Safety branch: non-resolving player picks info set
      // Resolving player still can't distinguish from initial chance
      if (player == resolving) {
        return "ses_start";
      } else {
        return "ses_safe:choosing";
      }

    case Phase::kExploitChanceInfoSet:
      // Exploit branch: chance node
      // Resolving player still sees "ses_start" (no info gained)
      // Non-resolving player sees they're in exploit branch
      if (player == resolving) {
        return "ses_start";
      } else {
        return "ses_exploit:chance";
      }

    case Phase::kChanceWithinInfoSet:
      // Both branches arrive here; resolving player sees "ses_start"
      // Non-resolving player knows which branch and which info set
      if (player == resolving) {
        return "ses_start";
      } else {
        if (chosen_branch_ == 0) {
          // Safety branch: non-resolving player chose the info set
          return absl::StrCat("ses_safe:within:", chosen_info_state_);
        } else {
          // Exploit branch: chance chose the info set
          return absl::StrCat("ses_exploit:within:", chosen_info_state_);
        }
      }

    case Phase::kSubgame:
      // BOTH branches share the same subgame info state prefix
      // This is the crucial SES property: resolving player can't distinguish
      // branches, and non-resolving player is forced to play the same strategy
      return absl::StrCat("ses_F:subgame:",
                          state_->InformationStateString(player));

    case Phase::kTerminal:
      return absl::StrCat("ses_F:subgame:",
                          state_->InformationStateString(player));
  }
  SpielFatalError("Unknown phase in SESGadgetState::InformationStateString");
}

std::string SESGadgetState::ObservationString(Player player) const {
  return InformationStateString(player);
}

std::string SESGadgetState::ToString() const {
  std::string result = "SESGadgetState(";
  switch (phase_) {
    case Phase::kInitialChance:
      result += "phase=InitialChance";
      break;
    case Phase::kSafeInfoSetChoice:
      result += "phase=SafeInfoSetChoice";
      break;
    case Phase::kExploitChanceInfoSet:
      result += "phase=ExploitChanceInfoSet";
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
                              ", root=", selected_root_idx_);
      break;
  }
  result += ")";
  return result;
}

std::unique_ptr<State> SESGadgetState::Clone() const {
  return std::make_unique<SESGadgetState>(*this);
}

std::vector<std::pair<Action, double>> SESGadgetState::ChanceOutcomes() const {
  const auto* ses_game = GetSESGame();

  switch (phase_) {
    case Phase::kInitialChance: {
      double alpha = ses_game->Alpha();
      return {{0, 1.0 - alpha}, {1, alpha}};
    }

    case Phase::kExploitChanceInfoSet:
      return ses_game->ExploitChanceOutcomes();

    case Phase::kChanceWithinInfoSet: {
      const auto& roots = ses_game->RootsForInfoSet(chosen_info_set_idx_);
      double reach_sum = ses_game->GetInfoSetReachSum(chosen_info_state_);

      std::vector<std::pair<Action, double>> outcomes;
      for (int i = 0; i < static_cast<int>(roots.size()); ++i) {
        double prob = ses_game->GetRootReachProbability(roots[i]) / reach_sum;
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

void SESGadgetState::DoApplyAction(Action action_id) {
  const auto* ses_game = GetSESGame();

  switch (phase_) {
    case Phase::kInitialChance:
      // Chance picks branch: 0=safety, 1=exploit
      chosen_branch_ = action_id;
      if (chosen_branch_ == 0) {
        // Safety branch: non-resolving player picks info set
        phase_ = Phase::kSafeInfoSetChoice;
      } else {
        // Exploit branch: chance picks info set by model reach
        phase_ = Phase::kExploitChanceInfoSet;
      }
      break;

    case Phase::kSafeInfoSetChoice:
      // Non-resolving player selects which info set to challenge
      chosen_info_set_idx_ = action_id;
      chosen_info_state_ = ses_game->InfoSetForAction(action_id);
      phase_ = Phase::kChanceWithinInfoSet;
      break;

    case Phase::kExploitChanceInfoSet:
      // Chance selects info set weighted by model reach
      chosen_info_set_idx_ = action_id;
      chosen_info_state_ = ses_game->InfoSetForAction(action_id);
      phase_ = Phase::kChanceWithinInfoSet;
      break;

    case Phase::kChanceWithinInfoSet: {
      // Chance selects which root state within the chosen info set
      const auto& roots = ses_game->RootsForInfoSet(chosen_info_set_idx_);
      selected_root_idx_ = roots[action_id];
      state_ = ses_game->CloneRootState(selected_root_idx_);
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

std::shared_ptr<const SESGadgetGame> CreateSESGadgetGame(
    std::shared_ptr<const Game> game,
    std::vector<SubgameRoot> subgame_roots,
    Player resolving_player,
    std::unordered_map<std::string, double> counterfactual_values,
    double alpha,
    std::unordered_map<std::string, double> model_info_set_reach) {
  return std::make_shared<SESGadgetGame>(
      game, std::move(subgame_roots), resolving_player,
      std::move(counterfactual_values), alpha, std::move(model_info_set_reach));
}

// =============================================================================
// ResolveWithSESGadget
// =============================================================================

std::shared_ptr<TabularPolicy> ResolveWithSESGadget(
    const SubgameDecomposition& decomp,
    const TabularPolicy& trunk_policy,
    double alpha,
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

    for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
      auto info_per_player = CollectSubgameInfoStatesPerPlayer(roots);

      // Build SubgameRoots for SES:
      // - info_state_string = NON-resolving player's info state (for grouping)
      // - reach_prob = resolving player's reach (π_{-nonres} = π_res * π_c)
      // We use BuildSubgameRoots with non_res as the "resolving_player" param
      // to get non_res's info state string, and decomp.reach_probs[res] for
      // the resolving player's reach.
      auto ses_roots =
          BuildSubgameRoots(roots, non_res, decomp.reach_probs[res]);
      if (ses_roots.empty()) continue;

      // Compute model info set reach: p̂(I) for each info set (grouped by
      // non-resolving player's info state)
      // Collect all root pointers for reach computation
      std::vector<const State*> root_ptrs;
      for (const auto& root : roots) root_ptrs.push_back(root.get());

      // Compute model's reach to each root state
      auto model_reach = ComputeReachProbabilities(
          *decomp.game, opponent_model, non_res, root_ptrs);

      // Aggregate by info set (non-resolving player's info state)
      std::unordered_map<std::string, double> model_info_set_reach;
      for (const auto& root : roots) {
        std::string non_res_is = root->InformationStateString(non_res);
        auto it = model_reach.find(root->HistoryString());
        if (it != model_reach.end()) {
          model_info_set_reach[non_res_is] += it->second;
        }
      }

      // CFVs for the NON-resolving player (keyed by non-resolving's info state)
      // We use decomp.cfvs[non_res] which is the CFV for the non-resolving
      // player. However, these are keyed by the non_res player's info state.
      // We pass them directly (same keys as info_state_string in ses_roots).
      auto ses_game = CreateSESGadgetGame(
          decomp.game, std::move(ses_roots), res, decomp.cfvs[non_res],
          alpha, model_info_set_reach);

      algorithms::CFRSolverBase solver(*ses_game, true, true, true);
      for (int i = 0; i < cfr_iterations; ++i) {
        solver.EvaluateAndUpdatePolicy();
      }

      // In SES, the NON-resolving player has artificial actions (info set
      // choice in safety branch), so their strategy is distorted. We extract
      // the RESOLVING player's strategy which only acts in the subgame.
      TabularPolicy gadget_policy = solver.TabularAveragePolicy();
      const std::string prefix = "ses_F:subgame:";
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
