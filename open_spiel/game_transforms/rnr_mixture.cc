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

#include "open_spiel/game_transforms/rnr_mixture.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {

// =============================================================================
// RNRMixtureGame
// =============================================================================

RNRMixtureGame::RNRMixtureGame(
    std::shared_ptr<const Game> free_game,
    std::shared_ptr<const Game> fixed_game,
    Player target_player,
    std::shared_ptr<const Policy> fixed_opponent_policy,
    double p,
    bool lock_opponent_in_fixed_branch,
    double fixed_branch_utility_multiplier,
    std::function<std::string(const State&, Player)> subgame_is_canonicalizer)
    : Game(GameType{
          "rnr_mixture",
          "RNR Mixture Game",
          GameType::Dynamics::kSequential,
          GameType::ChanceMode::kExplicitStochastic,
          GameType::Information::kImperfectInformation,
          GameType::Utility::kZeroSum,
          GameType::RewardModel::kTerminal,
          /*max_num_players=*/2,
          /*min_num_players=*/2,
          /*provides_information_state_string=*/true,
          /*provides_information_state_tensor=*/false,
          /*provides_observation_string=*/false,
          /*provides_observation_tensor=*/false,
          /*parameter_specification=*/{}},
          {}),
      free_game_(std::move(free_game)),
      fixed_game_(std::move(fixed_game)),
      target_player_(target_player),
      fixed_opponent_policy_(std::move(fixed_opponent_policy)),
      p_(p),
      lock_opponent_in_fixed_branch_(lock_opponent_in_fixed_branch),
      fixed_branch_utility_multiplier_(fixed_branch_utility_multiplier),
      subgame_is_canonicalizer_(std::move(subgame_is_canonicalizer)) {
  SPIEL_CHECK_TRUE(free_game_ != nullptr);
  SPIEL_CHECK_TRUE(fixed_game_ != nullptr);
  SPIEL_CHECK_TRUE(fixed_opponent_policy_ != nullptr);
  SPIEL_CHECK_GE(p_, 0.0);
  SPIEL_CHECK_LE(p_, 1.0);
  SPIEL_CHECK_GT(fixed_branch_utility_multiplier_, 0.0);
}

int RNRMixtureGame::NumDistinctActions() const {
  return std::max(free_game_->NumDistinctActions(),
                  fixed_game_->NumDistinctActions());
}

double RNRMixtureGame::MinUtility() const {
  return std::min(free_game_->MinUtility(), fixed_game_->MinUtility());
}

double RNRMixtureGame::MaxUtility() const {
  return std::max(free_game_->MaxUtility(), fixed_game_->MaxUtility());
}

int RNRMixtureGame::MaxGameLength() const {
  return 1 + std::max(free_game_->MaxGameLength(),
                      fixed_game_->MaxGameLength());
}

int RNRMixtureGame::MaxChanceOutcomes() const {
  // Root: 2 outcomes (free or fixed).
  // Fixed branch with lock=true: opponent nodes become chance with up to
  //   NumDistinctActions outcomes.
  // Inner games' own chance outcomes.
  return std::max({2,
                   free_game_->MaxChanceOutcomes(),
                   fixed_game_->MaxChanceOutcomes(),
                   fixed_game_->NumDistinctActions()});
}

std::unique_ptr<State> RNRMixtureGame::NewInitialState() const {
  return std::make_unique<RNRMixtureState>(shared_from_this());
}

// =============================================================================
// RNRMixtureState
// =============================================================================

RNRMixtureState::RNRMixtureState(std::shared_ptr<const Game> game)
    : State(game),
      branch_(RNRBranch::kRoot),
      inner_state_(nullptr) {}

RNRMixtureState::RNRMixtureState(const RNRMixtureState& other)
    : State(other.game_),
      branch_(other.branch_),
      inner_state_(other.inner_state_ ? other.inner_state_->Clone() : nullptr) {
  history_ = other.history_;
}

const RNRMixtureGame* RNRMixtureState::GetMixtureGame() const {
  return down_cast<const RNRMixtureGame*>(game_.get());
}

Player RNRMixtureState::CurrentPlayer() const {
  if (branch_ == RNRBranch::kRoot) {
    return kChancePlayerId;
  }
  if (IsTerminal()) return kTerminalPlayerId;

  const RNRMixtureGame* g = GetMixtureGame();
  Player inner_player = inner_state_->CurrentPlayer();

  if (branch_ == RNRBranch::kFixed && g->LockOpponentInFixedBranch()) {
    // In fixed branch with lock=true: opponent's decision nodes become chance.
    if (inner_player == g->OpponentPlayer()) {
      return kChancePlayerId;
    }
  }

  return inner_player;
}

std::vector<Action> RNRMixtureState::LegalActions() const {
  if (branch_ == RNRBranch::kRoot) {
    return {RNRMixtureGame::kFreeBranchAction, RNRMixtureGame::kFixedBranchAction};
  }
  if (IsTerminal()) return {};

  const RNRMixtureGame* g = GetMixtureGame();

  if (branch_ == RNRBranch::kFixed && g->LockOpponentInFixedBranch()) {
    Player inner_player = inner_state_->CurrentPlayer();
    if (inner_player == g->OpponentPlayer()) {
      // Chance takes over: enumerate actions for opponent
      // We'll use the same action indices as the inner game
      return inner_state_->LegalActions();
    }
  }

  return inner_state_->LegalActions();
}

std::string RNRMixtureState::ActionToString(Player player, Action action) const {
  if (branch_ == RNRBranch::kRoot) {
    return action == RNRMixtureGame::kFreeBranchAction ? "free" : "fixed";
  }
  return inner_state_->ActionToString(player, action);
}

bool RNRMixtureState::IsTerminal() const {
  if (branch_ == RNRBranch::kRoot) return false;
  return inner_state_->IsTerminal();
}

std::vector<double> RNRMixtureState::Returns() const {
  if (!IsTerminal()) {
    return std::vector<double>(game_->NumPlayers(), 0.0);
  }
  // The (1-p)/p weighting is handled by the root chance node's probabilities,
  // so we just return the inner terminal's utilities directly.
  std::vector<double> returns = inner_state_->Returns();
  if (branch_ == RNRBranch::kFixed) {
    const RNRMixtureGame* g = GetMixtureGame();
    for (double& v : returns) v *= g->FixedBranchUtilityMultiplier();
  }
  return returns;
}

std::string RNRMixtureState::InformationStateString(Player player) const {
  if (branch_ == RNRBranch::kRoot) {
    // At root, no player acts — shouldn't be called for non-chance player.
    // Return a dummy string.
    return "rnr_root";
  }

  const RNRMixtureGame* g = GetMixtureGame();

  if (player == g->TargetPlayer()) {
    // For the target player: try to canonicalize to original-game info state.
    // The canonicalizer strips the gadget/unsafe prefix.
    std::string canon = g->Canonicalize(*inner_state_, player);
    if (!canon.empty()) {
      // In-subgame node: use canonical original-game info state (shared).
      return canon;
    }
    // Gadget-specific entry node (outside subgame core): branch-local.
    if (branch_ == RNRBranch::kFree) {
      return absl::StrCat("rr_free_tgt:", inner_state_->InformationStateString(player));
    } else {
      return absl::StrCat("rr_fixed_tgt:", inner_state_->InformationStateString(player));
    }
  } else {
    // Opponent: always branch-local.
    if (branch_ == RNRBranch::kFree) {
      return absl::StrCat("rr_free:", inner_state_->InformationStateString(player));
    } else {
      // Fixed branch.
      if (g->LockOpponentInFixedBranch()) {
        // Opponent doesn't act here — but InformationStateString may still be
        // called by solvers traversing the tree. Return a fixed-branch tag.
        return absl::StrCat("rr_fixed_locked:", inner_state_->InformationStateString(player));
      } else {
        // Opponent acts freely with branch-local info states.
        // Strip the "unsafe:subgame:" prefix to get original IS, then tag.
        std::string inner_is = inner_state_->InformationStateString(player);
        // Try to canonicalize (strips "unsafe:subgame:" prefix)
        std::string canon_opp = g->Canonicalize(*inner_state_, player);
        if (!canon_opp.empty()) {
          return absl::StrCat("rr_fixed:", canon_opp);
        }
        return absl::StrCat("rr_fixed:", inner_is);
      }
    }
  }
}

std::string RNRMixtureState::ObservationString(Player player) const {
  return InformationStateString(player);
}

std::string RNRMixtureState::ToString() const {
  if (branch_ == RNRBranch::kRoot) return "RNRMixture(root)";
  std::string branch_str = (branch_ == RNRBranch::kFree) ? "free" : "fixed";
  return absl::StrCat("RNRMixture(", branch_str, ",", inner_state_->ToString(), ")");
}

std::unique_ptr<State> RNRMixtureState::Clone() const {
  return std::make_unique<RNRMixtureState>(*this);
}

std::vector<std::pair<Action, double>> RNRMixtureState::ChanceOutcomes() const {
  const RNRMixtureGame* g = GetMixtureGame();

  if (branch_ == RNRBranch::kRoot) {
    // Root chance: pick free (1-p) or fixed (p).
    std::vector<std::pair<Action, double>> outcomes;
    if (g->P() < 1.0) {
      outcomes.push_back({RNRMixtureGame::kFreeBranchAction, 1.0 - g->P()});
    }
    if (g->P() > 0.0) {
      outcomes.push_back({RNRMixtureGame::kFixedBranchAction, g->P()});
    }
    // Edge cases: p=0 → only free; p=1 → only fixed.
    if (outcomes.empty()) {
      // Should not happen if p in [0,1].
      SpielFatalError("RNRMixtureGame: p is neither 0 nor 1, but outcomes empty");
    }
    return outcomes;
  }

  if (branch_ == RNRBranch::kFixed && g->LockOpponentInFixedBranch()) {
    Player inner_player = inner_state_->CurrentPlayer();
    if (inner_player == g->OpponentPlayer()) {
      // Opponent node in fixed branch: sample from σ^fix(orig_is).
      // The inner state's IS includes "unsafe:subgame:" prefix — strip it.
      std::string inner_is = inner_state_->InformationStateString(g->OpponentPlayer());
      // Try to get original IS by canonicalizing.
      std::string orig_is = g->Canonicalize(*inner_state_, g->OpponentPlayer());
      if (orig_is.empty()) {
        // Fallback: strip "unsafe:subgame:" prefix manually.
        const std::string prefix = "unsafe:subgame:";
        if (inner_is.compare(0, prefix.size(), prefix) == 0) {
          orig_is = inner_is.substr(prefix.size());
        } else {
          orig_is = inner_is;  // use as-is
        }
      }
      auto ap = g->FixedOpponentPolicy().GetStatePolicy(orig_is);
      if (ap.empty()) {
        // Fall back to uniform over legal actions.
        auto legal = inner_state_->LegalActions();
        if (legal.empty()) return {};
        double prob = 1.0 / static_cast<double>(legal.size());
        std::vector<std::pair<Action, double>> result;
        result.reserve(legal.size());
        for (Action a : legal) result.push_back({a, prob});
        return result;
      }
      return ap;
    }
  }

  // Delegate to inner state's chance outcomes.
  return inner_state_->ChanceOutcomes();
}

void RNRMixtureState::DoApplyAction(Action action) {
  const RNRMixtureGame* g = GetMixtureGame();

  if (branch_ == RNRBranch::kRoot) {
    if (action == RNRMixtureGame::kFreeBranchAction) {
      branch_ = RNRBranch::kFree;
      inner_state_ = g->FreeGame().NewInitialState();
    } else {
      branch_ = RNRBranch::kFixed;
      inner_state_ = g->FixedGame().NewInitialState();
    }
    return;
  }

  inner_state_->ApplyAction(action);
}

// =============================================================================
// Factory and helpers
// =============================================================================

std::shared_ptr<const RNRMixtureGame> CreateRNRMixtureGame(
    std::shared_ptr<const Game> free_game,
    std::shared_ptr<const Game> fixed_game,
    Player target_player,
    std::shared_ptr<const Policy> fixed_opponent_policy,
    double p,
    bool lock_opponent_in_fixed_branch,
    double fixed_branch_utility_multiplier,
    std::function<std::string(const State&, Player)> subgame_is_canonicalizer) {
  return std::make_shared<RNRMixtureGame>(
      std::move(free_game),
      std::move(fixed_game),
      target_player,
      std::move(fixed_opponent_policy),
      p,
      lock_opponent_in_fixed_branch,
      fixed_branch_utility_multiplier,
      std::move(subgame_is_canonicalizer));
}

std::function<std::string(const State&, Player)>
MakeSubgameISCanonicalizer(const std::string& free_prefix,
                           const std::string& fixed_prefix) {
  return [free_prefix, fixed_prefix](const State& inner_state, Player player)
      -> std::string {
    std::string is = inner_state.InformationStateString(player);
    // Try free prefix first.
    if (!free_prefix.empty() &&
        is.compare(0, free_prefix.size(), free_prefix) == 0) {
      return is.substr(free_prefix.size());
    }
    // Try fixed prefix.
    if (!fixed_prefix.empty() &&
        is.compare(0, fixed_prefix.size(), fixed_prefix) == 0) {
      return is.substr(fixed_prefix.size());
    }
    // Not in subgame core.
    return "";
  };
}

}  // namespace open_spiel
