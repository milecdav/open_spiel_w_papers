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

#include "open_spiel/game_transforms/resolving_gadget.h"

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

// Game type for the gadget transformation
const GameType kGadgetGameType{
    /*short_name=*/"resolving_gadget",
    /*long_name=*/"Resolving Gadget Game",
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
  GameType new_type = kGadgetGameType;
  new_type.long_name = "Resolving Gadget " + type.long_name;
  new_type.utility = GameType::Utility::kZeroSum;  // Gadget preserves zero-sum
  new_type.max_num_players = 2;
  new_type.min_num_players = 2;
  new_type.provides_information_state_string =
      type.provides_information_state_string;
  new_type.provides_observation_string = type.provides_observation_string;
  return new_type;
}

}  // namespace

// =============================================================================
// GadgetGame implementation
// =============================================================================

GadgetGame::GadgetGame(
    std::shared_ptr<const Game> game,
    std::vector<SubgameRoot> subgame_roots,
    Player adversary_player,
    std::unordered_map<std::string, double> counterfactual_values)
    : WrappedGame(game, ConvertType(game->GetType()), game->GetParameters()),
      subgame_roots_(std::move(subgame_roots)),
      adversary_player_(adversary_player),
      counterfactual_values_(std::move(counterfactual_values)),
      k_(0.0) {
  SPIEL_CHECK_GT(subgame_roots_.size(), 0);
  SPIEL_CHECK_GE(adversary_player_, 0);
  SPIEL_CHECK_LT(adversary_player_, 2);
  SPIEL_CHECK_EQ(game->NumPlayers(), 2);

  // Compute normalization constant k = Σ_r π_{res}(r)
  for (const auto& root : subgame_roots_) {
    k_ += root.reach_prob;
  }
  SPIEL_CHECK_GT(k_, 0.0);

  // Compute T-action payoffs
  ComputeTerminatePayoffs();
}

void GadgetGame::ComputeTerminatePayoffs() {
  // First, accumulate reach probs per info state
  for (const auto& root : subgame_roots_) {
    info_state_reach_sums_[root.info_state_string] += root.reach_prob;
  }

  // Then compute T payoffs: T_payoff = k * v^R(I) / Σ_{h∈I} π_{res}(h)
  for (const auto& [info_state, reach_sum] : info_state_reach_sums_) {
    auto it = counterfactual_values_.find(info_state);
    double cf_value = 0.0;
    if (it != counterfactual_values_.end()) {
      cf_value = it->second;
    }
    // The T payoff for the non-resolving (adversary) player
    terminate_payoffs_[info_state] = k_ * cf_value / reach_sum;
  }
}

std::unique_ptr<State> GadgetGame::NewInitialState() const {
  return std::make_unique<GadgetState>(shared_from_this());
}

int GadgetGame::NumDistinctActions() const {
  // Max of: 2 (T/F actions) and the original game's actions
  return std::max(2, game_->NumDistinctActions());
}

int GadgetGame::MaxGameLength() const {
  // Chance (1) + T/F choice (1) + original subgame length
  return 2 + game_->MaxGameLength();
}

double GadgetGame::GetTerminatePayoff(const std::string& info_state) const {
  auto it = terminate_payoffs_.find(info_state);
  if (it != terminate_payoffs_.end()) {
    return it->second;
  }
  // If not found, return 0 (shouldn't happen in well-formed gadgets)
  throw std::runtime_error("Terminate payoff not found for info state: " + info_state);
}

double GadgetGame::GetRootReachProbability(int root_idx) const {
  SPIEL_CHECK_GE(root_idx, 0);
  SPIEL_CHECK_LT(root_idx, subgame_roots_.size());
  return subgame_roots_[root_idx].reach_prob;
}

const std::string& GadgetGame::GetRootInfoState(int root_idx) const {
  SPIEL_CHECK_GE(root_idx, 0);
  SPIEL_CHECK_LT(root_idx, subgame_roots_.size());
  return subgame_roots_[root_idx].info_state_string;
}

std::unique_ptr<State> GadgetGame::CloneRootState(int root_idx) const {
  SPIEL_CHECK_GE(root_idx, 0);
  SPIEL_CHECK_LT(root_idx, subgame_roots_.size());
  return subgame_roots_[root_idx].state->Clone();
}

double GadgetGame::MinUtility() const {
  return k_ * game_->MinUtility();
}

double GadgetGame::MaxUtility() const {
  return k_ * game_->MaxUtility();
}

// =============================================================================
// GadgetState implementation
// =============================================================================

GadgetState::GadgetState(std::shared_ptr<const Game> game)
    : WrappedState(game, nullptr),  // No wrapped state initially
      phase_(Phase::kChance),
      selected_root_idx_(-1),
      chose_terminate_(false),
      root_info_state_("") {}

GadgetState::GadgetState(const GadgetState& other)
    : WrappedState(other.game_, other.state_ ? other.state_->Clone() : nullptr),
      phase_(other.phase_),
      selected_root_idx_(other.selected_root_idx_),
      chose_terminate_(other.chose_terminate_),
      root_info_state_(other.root_info_state_) {
  // Copy history
  history_ = other.history_;
}

const GadgetGame* GadgetState::GetGadgetGame() const {
  return down_cast<const GadgetGame*>(game_.get());
}

Player GadgetState::CurrentPlayer() const {
  switch (phase_) {
    case Phase::kChance:
      return kChancePlayerId;
    case Phase::kGadgetChoice:
      // The adversary (non-resolving player) chooses T or F
      return GetGadgetGame()->NonResolvingPlayer();
    case Phase::kSubgame:
      return state_->CurrentPlayer();
    case Phase::kTerminal:
      return kTerminalPlayerId;
  }
  SpielFatalError("Unknown phase in GadgetState::CurrentPlayer");
}

std::vector<Action> GadgetState::LegalActions() const {
  switch (phase_) {
    case Phase::kChance:
      // Chance actions are indices into subgame_roots
      {
        std::vector<Action> actions;
        for (int i = 0; i < GetGadgetGame()->NumSubgameRoots(); ++i) {
          actions.push_back(i);
        }
        return actions;
      }
    case Phase::kGadgetChoice:
      // T or F
      return {GadgetGame::kTerminateAction, GadgetGame::kFollowAction};
    case Phase::kSubgame:
      return state_->LegalActions();
    case Phase::kTerminal:
      return {};
  }
  SpielFatalError("Unknown phase in GadgetState::LegalActions");
}

std::vector<Action> GadgetState::LegalActions(Player player) const {
  if (phase_ == Phase::kSubgame) {
    return state_->LegalActions(player);
  }
  if (player == CurrentPlayer()) {
    return LegalActions();
  }
  return {};
}

std::string GadgetState::ActionToString(Player player, Action action_id) const {
  switch (phase_) {
    case Phase::kChance:
      return absl::StrCat("root_", action_id);
    case Phase::kGadgetChoice:
      if (action_id == GadgetGame::kTerminateAction) {
        return "T";
      } else {
        return "F";
      }
    case Phase::kSubgame:
      return state_->ActionToString(player, action_id);
    case Phase::kTerminal:
      return "";
  }
  SpielFatalError("Unknown phase in GadgetState::ActionToString");
}

bool GadgetState::IsTerminal() const {
  return phase_ == Phase::kTerminal;
}

std::vector<double> GadgetState::Returns() const {
  const GadgetGame* gadget_game = GetGadgetGame();
  std::vector<double> returns(2, 0.0);

  if (phase_ != Phase::kTerminal) {
    return returns;
  }

  if (chose_terminate_) {
    // T was chosen: non-resolving (adversary) player gets their original CF value (scaled)
    double t_payoff = gadget_game->GetTerminatePayoff(root_info_state_);
    returns[gadget_game->NonResolvingPlayer()] = t_payoff;
    returns[gadget_game->ResolvingPlayer()] = -t_payoff;  // Zero-sum
  } else {
    // F was chosen: return the subgame returns (scaled by k)
    auto subgame_returns = state_->Returns();
    double k = gadget_game->NormalizationConstant();
    for (int p = 0; p < 2; ++p) {
      returns[p] = k * subgame_returns[p];
    }
  }

  return returns;
}

std::vector<double> GadgetState::Rewards() const {
  // Terminal rewards model
  if (IsTerminal()) {
    return Returns();
  }
  return std::vector<double>(2, 0.0);
}

std::string GadgetState::InformationStateString(Player player) const {
  const GadgetGame* gadget_game = GetGadgetGame();

  switch (phase_) {
    case Phase::kChance:
      // Before any action, both players have empty info
      return "gadget_start";

    case Phase::kGadgetChoice:
      // The non-resolving (adversary) player sees their info state at the root
      if (player == gadget_game->NonResolvingPlayer()) {
        return absl::StrCat("gadget_choice:", root_info_state_);
      } else {
        // Resolving player doesn't know which root was selected
        // but knows we're at the gadget choice
        return "gadget_choice:opponent_choosing";
      }

    case Phase::kSubgame:
      // In subgame, both players use the same format with underlying info state
      // The current info state already contains all information needed
      return absl::StrCat("gadget_F:subgame:",
                          state_->InformationStateString(player));

    case Phase::kTerminal:
      if (chose_terminate_) {
        if (player == gadget_game->NonResolvingPlayer()) {
          return absl::StrCat("gadget_T:", root_info_state_);
        } else {
          return "gadget_T:opponent_terminated";
        }
      } else {
        // Terminal after playing through subgame (F was chosen)
        return absl::StrCat("gadget_F:subgame:",
                            state_->InformationStateString(player));
      }
  }
  SpielFatalError("Unknown phase in GadgetState::InformationStateString");
}

std::string GadgetState::ObservationString(Player player) const {
  // For simplicity, same as information state
  return InformationStateString(player);
}

std::string GadgetState::ToString() const {
  std::string result = "GadgetState(";
  switch (phase_) {
    case Phase::kChance:
      result += "phase=Chance";
      break;
    case Phase::kGadgetChoice:
      result += absl::StrCat("phase=GadgetChoice, root=", selected_root_idx_,
                              ", info=", root_info_state_);
      break;
    case Phase::kSubgame:
      result += absl::StrCat("phase=Subgame, root=", selected_root_idx_,
                              ", state=", state_->ToString());
      break;
    case Phase::kTerminal:
      result += absl::StrCat("phase=Terminal, T=", chose_terminate_);
      break;
  }
  result += ")";
  return result;
}

std::unique_ptr<State> GadgetState::Clone() const {
  return std::make_unique<GadgetState>(*this);
}

std::vector<std::pair<Action, double>> GadgetState::ChanceOutcomes() const {
  if (phase_ == Phase::kChance) {
    // Initial gadget chance node
    const GadgetGame* gadget_game = GetGadgetGame();
    double k = gadget_game->NormalizationConstant();

    std::vector<std::pair<Action, double>> outcomes;
    for (int i = 0; i < gadget_game->NumSubgameRoots(); ++i) {
      double prob = gadget_game->GetRootReachProbability(i) / k;
      outcomes.push_back({i, prob});
    }
    return outcomes;
  } else if (phase_ == Phase::kSubgame && state_->IsChanceNode()) {
    // Delegate to underlying subgame chance node
    return state_->ChanceOutcomes();
  }
  return {};
}

void GadgetState::DoApplyAction(Action action_id) {
  const GadgetGame* gadget_game = GetGadgetGame();

  switch (phase_) {
    case Phase::kChance:
      // Chance selects which root state to use
      selected_root_idx_ = action_id;
      root_info_state_ = gadget_game->GetRootInfoState(action_id);
      state_ = gadget_game->CloneRootState(action_id);
      phase_ = Phase::kGadgetChoice;
      break;

    case Phase::kGadgetChoice:
      // Non-resolving (adversary) player chooses T or F
      if (action_id == GadgetGame::kTerminateAction) {
        chose_terminate_ = true;
        phase_ = Phase::kTerminal;
      } else {
        chose_terminate_ = false;
        // Check if subgame is already terminal
        if (state_->IsTerminal()) {
          phase_ = Phase::kTerminal;
        } else {
          phase_ = Phase::kSubgame;
        }
      }
      break;

    case Phase::kSubgame:
      // Apply action in the underlying subgame
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

std::shared_ptr<const GadgetGame> CreateGadgetGame(
    std::shared_ptr<const Game> game,
    std::vector<SubgameRoot> subgame_roots,
    Player adversary_player,
    std::unordered_map<std::string, double> counterfactual_values) {
  return std::make_shared<GadgetGame>(
      game, std::move(subgame_roots), adversary_player,
      std::move(counterfactual_values));
}

// =============================================================================
// ResolveWithGadget
// =============================================================================

std::shared_ptr<TabularPolicy> ResolveWithGadget(
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

      // adversary_player = non_res (has T/F artificial actions)
      auto gadget = CreateGadgetGame(
          decomp.game, std::move(gadget_roots), non_res, decomp.cfvs[non_res]);
      algorithms::CFRSolverBase solver(*gadget, true, true, true);
      for (int i = 0; i < cfr_iterations; ++i) {
        solver.EvaluateAndUpdatePolicy();
      }

      // Extract the resolving player's strategy from the gadget
      TabularPolicy gadget_policy = solver.TabularAveragePolicy();
      const std::string prefix = "gadget_F:subgame:";
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
