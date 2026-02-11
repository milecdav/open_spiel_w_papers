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

#include "open_spiel/game_transforms/continual_resolving.h"

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/algorithms/rnr.h"
#include "open_spiel/game_transforms/max_margin_gadget.h"
#include "open_spiel/game_transforms/matrix_valued_states.h"
#include "open_spiel/game_transforms/resolving_gadget.h"
#include "open_spiel/game_transforms/subgame_utils.h"
#include "open_spiel/game_transforms/unsafe_subgame.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"

namespace open_spiel {

namespace {

// Tabularize a policy by traversing the game tree.
// Uses the State-based GetStatePolicy (works for UniformPolicy etc.)
TabularPolicy TabularizePolicy(const Game& game, const Policy& policy) {
  TabularPolicy result;
  std::function<void(const State&)> traverse = [&](const State& state) {
    if (state.IsTerminal()) return;
    if (state.IsChanceNode()) {
      for (const auto& [a, p] : state.ChanceOutcomes()) {
        auto child = state.Clone();
        child->ApplyAction(a);
        traverse(*child);
      }
    } else {
      Player pl = state.CurrentPlayer();
      auto ap = policy.GetStatePolicy(state, pl);
      if (!ap.empty()) {
        result.SetStatePolicy(state.InformationStateString(pl), ap);
      }
      for (Action a : state.LegalActions()) {
        auto child = state.Clone();
        child->ApplyAction(a);
        traverse(*child);
      }
    }
  };
  traverse(*game.NewInitialState());
  return result;
}

}  // namespace

// ============================================================================
// GadgetPolicyWrapper
// ============================================================================

GadgetPolicyWrapper::GadgetPolicyWrapper(const Policy* underlying,
                                         std::string subgame_prefix,
                                         int follow_action,
                                         int num_gadget_choice_actions)
    : underlying_(underlying),
      subgame_prefix_(std::move(subgame_prefix)),
      follow_action_(follow_action),
      num_gadget_choice_actions_(num_gadget_choice_actions) {}

ActionsAndProbs GadgetPolicyWrapper::GetStatePolicy(
    const std::string& info_state) const {
  // Subgame info states: strip prefix, delegate to underlying
  if (info_state.compare(0, subgame_prefix_.length(), subgame_prefix_) == 0) {
    std::string orig = info_state.substr(subgame_prefix_.length());
    return underlying_->GetStatePolicy(orig);
  }

  // Resolving gadget T/F choice: always Follow
  if (follow_action_ >= 0 &&
      info_state.compare(0, 14, "gadget_choice:") == 0 &&
      info_state != "gadget_choice:opponent_choosing") {
    return {{GadgetGame::kTerminateAction, 0.0},
            {follow_action_, 1.0}};
  }

  // Max-margin info set choice: uniform over info sets
  if (num_gadget_choice_actions_ > 0 && info_state == "mm_start") {
    ActionsAndProbs result;
    double prob = 1.0 / num_gadget_choice_actions_;
    for (int i = 0; i < num_gadget_choice_actions_; ++i) {
      result.push_back({i, prob});
    }
    return result;
  }

  // Unknown info state - return empty (caller should use State-based version)
  return {};
}

ActionsAndProbs GadgetPolicyWrapper::GetStatePolicy(
    const State& state, Player player) const {
  std::string info_state = state.InformationStateString(player);

  // Try string-based lookup first
  auto result = GetStatePolicy(info_state);
  if (!result.empty()) return result;

  // Fall back to uniform over legal actions
  auto legal = state.LegalActions(player);
  if (legal.empty()) return {};
  double prob = 1.0 / legal.size();
  for (Action a : legal) {
    result.push_back({a, prob});
  }
  return result;
}

// ============================================================================
// ResolveSubgames
// ============================================================================

std::shared_ptr<TabularPolicy> ResolveSubgames(
    const SubgameDecomposition& decomp,
    const TabularPolicy& trunk_policy,
    const ResolvingConfig& config) {
  auto combined = std::make_shared<TabularPolicy>();

  // Copy trunk
  for (const auto& [player, info_states] : decomp.trunk_info_states) {
    for (const auto& is : info_states) {
      auto ap = trunk_policy.GetStatePolicy(is);
      if (!ap.empty()) combined->SetStatePolicy(is, ap);
    }
  }

  if (config.solver == SolverType::kCFR) {
    // Standard equilibrium solving
    if (config.gadget == GadgetType::kNone) {
      // Unsafe: solve once, extract both players' strategies
      for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
        std::vector<std::unique_ptr<State>> subgame_roots;
        std::vector<double> total_reaches;
        for (const auto& root : roots) {
          std::string hist = root->HistoryString();
          double r0 = 0, r1 = 0;
          auto it0 = decomp.reach_probs[0].find(hist);
          if (it0 != decomp.reach_probs[0].end()) r0 = it0->second;
          auto it1 = decomp.reach_probs[1].find(hist);
          if (it1 != decomp.reach_probs[1].end()) r1 = it1->second;
          double total = r0 * r1;
          if (total > 0) {
            subgame_roots.push_back(root->Clone());
            total_reaches.push_back(total);
          }
        }
        if (subgame_roots.empty()) continue;

        auto unsafe = CreateUnsafeSubgame(
            decomp.game, std::move(subgame_roots), std::move(total_reaches));
        algorithms::CFRSolverBase solver(*unsafe, true, true, true);
        for (int i = 0; i < config.cfr_iterations; ++i) {
          solver.EvaluateAndUpdatePolicy();
        }

        TabularPolicy policy = solver.TabularAveragePolicy();
        const std::string prefix = "unsafe:subgame:";
        for (const auto& [sub_is, ap] : policy.PolicyTable()) {
          if (sub_is.compare(0, prefix.length(), prefix) == 0) {
            combined->SetStatePolicy(sub_is.substr(prefix.length()), ap);
          }
        }
      }
    } else {
      // With gadget: solve per-player
      for (int non_res = 0; non_res < 2; ++non_res) {
        int res = 1 - non_res;
        for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
          auto info_per_player = CollectSubgameInfoStatesPerPlayer(roots);
          auto gadget_roots =
              BuildSubgameRoots(roots, res, decomp.reach_probs[non_res]);
          if (gadget_roots.empty()) continue;

          std::shared_ptr<const Game> gadget_game;
          std::string prefix;

          if (config.gadget == GadgetType::kResolving) {
            gadget_game = CreateGadgetGame(
                decomp.game, std::move(gadget_roots), res, decomp.cfvs[res]);
            prefix = "gadget_F:subgame:";
          } else {  // kMaxMargin
            gadget_game = CreateMaxMarginGadgetGame(
                decomp.game, std::move(gadget_roots), res, decomp.cfvs[res]);
            prefix = "mm_F:subgame:";
          }

          algorithms::CFRSolverBase solver(*gadget_game, true, true, true);
          for (int i = 0; i < config.cfr_iterations; ++i) {
            solver.EvaluateAndUpdatePolicy();
          }

          TabularPolicy policy = solver.TabularAveragePolicy();
          for (const auto& [sub_is, ap] : policy.PolicyTable()) {
            if (sub_is.compare(0, prefix.length(), prefix) == 0) {
              std::string orig = sub_is.substr(prefix.length());
              if (info_per_player[non_res].count(orig) > 0) {
                combined->SetStatePolicy(orig, ap);
              }
            }
          }
        }
      }
    }
  } else {
    // RNR-based solving
    SPIEL_CHECK_TRUE(config.opponent_model != nullptr);
    int target = config.target_player;
    int opponent = 1 - target;

    for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
      auto info_per_player = CollectSubgameInfoStatesPerPlayer(roots);

      std::shared_ptr<const Game> subgame_game;
      std::string prefix;
      int follow_action = -1;
      int num_choice_actions = 0;

      if (config.gadget == GadgetType::kResolving) {
        int res = opponent;
        auto gadget_roots =
            BuildSubgameRoots(roots, res, decomp.reach_probs[target]);
        if (gadget_roots.empty()) continue;
        subgame_game = CreateGadgetGame(
            decomp.game, std::move(gadget_roots), res, decomp.cfvs[res]);
        prefix = "gadget_F:subgame:";
        follow_action = GadgetGame::kFollowAction;
      } else if (config.gadget == GadgetType::kMaxMargin) {
        int res = opponent;
        auto gadget_roots =
            BuildSubgameRoots(roots, res, decomp.reach_probs[target]);
        if (gadget_roots.empty()) continue;
        auto mm_game = CreateMaxMarginGadgetGame(
            decomp.game, std::move(gadget_roots), res, decomp.cfvs[res]);
        num_choice_actions = mm_game->NumInfoSets();
        subgame_game = mm_game;
        prefix = "mm_F:subgame:";
      } else {  // kNone
        std::vector<std::unique_ptr<State>> subgame_roots;
        std::vector<double> total_reaches;
        for (const auto& root : roots) {
          std::string hist = root->HistoryString();
          double r0 = 0, r1 = 0;
          auto it0 = decomp.reach_probs[0].find(hist);
          if (it0 != decomp.reach_probs[0].end()) r0 = it0->second;
          auto it1 = decomp.reach_probs[1].find(hist);
          if (it1 != decomp.reach_probs[1].end()) r1 = it1->second;
          double total = r0 * r1;
          if (total > 0) {
            subgame_roots.push_back(root->Clone());
            total_reaches.push_back(total);
          }
        }
        if (subgame_roots.empty()) continue;
        subgame_game = CreateUnsafeSubgame(
            decomp.game, std::move(subgame_roots), std::move(total_reaches));
        prefix = "unsafe:subgame:";
      }

      // Create wrapper for opponent model
      GadgetPolicyWrapper wrapper(config.opponent_model, prefix,
                                  follow_action, num_choice_actions);

      // Solve with RNR
      algorithms::RNRSolver solver(*subgame_game, target, &wrapper, config.p,
                                   true, true, true);
      for (int i = 0; i < config.cfr_iterations; ++i) {
        solver.EvaluateAndUpdatePolicy();
      }

      // Extract target player's strategy
      TabularPolicy policy = solver.TabularAveragePolicy();
      for (const auto& [sub_is, ap] : policy.PolicyTable()) {
        if (sub_is.compare(0, prefix.length(), prefix) == 0) {
          std::string orig = sub_is.substr(prefix.length());
          if (info_per_player[target].count(orig) > 0) {
            combined->SetStatePolicy(orig, ap);
          }
        }
      }

      // Copy opponent's strategy from model
      for (const auto& is : info_per_player[opponent]) {
        auto ap = config.opponent_model->GetStatePolicy(is);
        if (!ap.empty()) combined->SetStatePolicy(is, ap);
      }
    }
  }

  return combined;
}

// ============================================================================
// ContinualResolve
// ============================================================================

std::shared_ptr<TabularPolicy> ContinualResolve(
    std::shared_ptr<const Game> game,
    const Policy& opponent_model,
    const ResolvingConfig& config,
    int depth,
    MVSGame::DepthMode depth_mode) {
  // Tabularize opponent model so string-based GetStatePolicy works
  TabularPolicy opponent_tabular = TabularizePolicy(*game, opponent_model);

  // Step 1: Create and solve MVS trunk
  auto mvs_game = CreateMVSGameWithSubtreePureStrategies(
      game, depth, depth_mode);
  algorithms::CFRSolverBase trunk_solver(*mvs_game, true, true, true);

  // Fix opponent's strategy in trunk when we have an opponent model
  auto trunk_is = (depth_mode == MVSGame::DepthMode::kRoundBased)
      ? CollectInfoStateStringsBeforeRound(*game, depth)
      : CollectInfoStateStringsBeforeDepth(*game, depth);
  int opponent = 1 - config.target_player;
  auto it = trunk_is.find(opponent);
  if (it != trunk_is.end() && !it->second.empty()) {
    auto opp_policy = std::make_shared<TabularPolicy>();
    for (const auto& is : it->second) {
      auto ap = opponent_tabular.GetStatePolicy(is);
      if (!ap.empty()) opp_policy->SetStatePolicy(is, ap);
    }
    trunk_solver.SetFixedPolicy(opp_policy, it->second);
  }

  for (int i = 0; i < config.cfr_iterations; ++i) {
    trunk_solver.EvaluateAndUpdatePolicy();
  }
  auto mvs_policy = trunk_solver.AveragePolicy();

  // Step 2: Extract trunk policy into original game's info state space
  TabularPolicy trunk;
  std::function<void(const State&, const State&)> copy_trunk =
      [&](const State& mvs_state, const State& orig_state) {
    if (orig_state.IsTerminal()) return;
    auto* mvs_s = dynamic_cast<const MVSStateWithSubtreePureStrategies*>(
        &mvs_state);
    if (mvs_s && mvs_s->GetPhase() != MVSState::Phase::kNormal) return;
    if (orig_state.IsChanceNode()) {
      for (const auto& [a, p] : orig_state.ChanceOutcomes()) {
        auto no = orig_state.Clone(); no->ApplyAction(a);
        auto nm = mvs_state.Clone(); nm->ApplyAction(a);
        copy_trunk(*nm, *no);
      }
    } else {
      Player pl = orig_state.CurrentPlayer();
      auto ap = mvs_policy->GetStatePolicy(mvs_state, pl);
      if (!ap.empty()) {
        trunk.SetStatePolicy(orig_state.InformationStateString(pl), ap);
      }
      for (Action a : orig_state.LegalActions()) {
        auto no = orig_state.Clone(); no->ApplyAction(a);
        auto nm = mvs_state.Clone(); nm->ApplyAction(a);
        copy_trunk(*nm, *no);
      }
    }
  };
  copy_trunk(*mvs_game->NewInitialState(), *game->NewInitialState());

  // Step 3: Build SubgameDecomposition
  SubgameDecomposition decomp;
  decomp.game = game;
  decomp.trunk_info_states = trunk_is;

  if (depth_mode == MVSGame::DepthMode::kRoundBased) {
    decomp.grouped_subgames = GroupStatesByPublicObservation(
        *game, CollectStatesAtRound(*game, depth));
  } else {
    decomp.grouped_subgames = GroupStatesByPublicObservation(
        *game, CollectStatesAtDepth(*game, depth));
  }

  decomp.reach_probs[0] = ExtractReachProbsFromMVS(*mvs_game, *mvs_policy, 0);
  decomp.reach_probs[1] = ExtractReachProbsFromMVS(*mvs_game, *mvs_policy, 1);
  decomp.cfvs[0] = ExtractCFVsFromMVSSolution(*mvs_game, *mvs_policy, 0);
  decomp.cfvs[1] = ExtractCFVsFromMVSSolution(*mvs_game, *mvs_policy, 1);

  // Step 4: Resolve subgames
  ResolvingConfig subgame_config = config;
  subgame_config.opponent_model = &opponent_tabular;

  return ResolveSubgames(decomp, trunk, subgame_config);
}

}  // namespace open_spiel
