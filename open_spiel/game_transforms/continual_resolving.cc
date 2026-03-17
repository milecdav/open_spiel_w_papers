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
// MVSOpponentPolicy
// ============================================================================

MVSOpponentPolicy::MVSOpponentPolicy(
    const Policy* underlying,
    std::unordered_map<std::string, ActionsAndProbs> portfolio_probs)
    : underlying_(underlying),
      portfolio_probs_(std::move(portfolio_probs)) {}

ActionsAndProbs MVSOpponentPolicy::GetStatePolicy(
    const std::string& info_state) const {
  // Check for MVS portfolio info states first
  auto it = portfolio_probs_.find(info_state);
  if (it != portfolio_probs_.end()) {
    return it->second;
  }
  // Delegate to underlying model for regular info states
  return underlying_->GetStatePolicy(info_state);
}

ActionsAndProbs MVSOpponentPolicy::GetStatePolicy(
    const State& state, Player player) const {
  std::string info_state = state.InformationStateString(player);
  auto result = GetStatePolicy(info_state);
  if (!result.empty()) return result;
  // Fall back to uniform
  auto legal = state.LegalActions(player);
  if (legal.empty()) return {};
  double prob = 1.0 / legal.size();
  for (Action a : legal) {
    result.push_back({a, prob});
  }
  return result;
}

// ============================================================================
// MVSModelEntryPolicy
// ============================================================================

MVSModelEntryPolicy::MVSModelEntryPolicy(
    const Policy* underlying,
    std::unordered_map<std::string, ActionsAndProbs> portfolio_probs)
    : underlying_(underlying),
      portfolio_probs_(std::move(portfolio_probs)) {}

ActionsAndProbs MVSModelEntryPolicy::GetStatePolicy(
    const std::string& info_state) const {
  // Check if this is an MVS portfolio choice info state for the opponent
  auto it = portfolio_probs_.find(info_state);
  if (it != portfolio_probs_.end()) {
    // Return full distribution with prob 1 on model entry, 0 elsewhere
    return it->second;
  }
  // Delegate to underlying model for regular info states
  return underlying_->GetStatePolicy(info_state);
}

ActionsAndProbs MVSModelEntryPolicy::GetStatePolicy(
    const State& state, Player player) const {
  std::string info_state = state.InformationStateString(player);
  auto result = GetStatePolicy(info_state);
  if (!result.empty()) return result;
  // Fall back to uniform
  auto legal = state.LegalActions(player);
  if (legal.empty()) return {};
  double prob = 1.0 / legal.size();
  for (Action a : legal) {
    result.push_back({a, prob});
  }
  return result;
}

std::unordered_map<std::string, ActionsAndProbs> PrecomputeModelActionIndices(
    const MVSGameWithSubtreePureStrategies& mvs_game,
    Player opponent) {
  std::unordered_map<std::string, ActionsAndProbs> result;

  // Determine which portfolio phase corresponds to the opponent
  // P0 selects at kPortfolioP1, P1 selects at kPortfolioP2
  auto opponent_phase = (opponent == 0)
      ? MVSState::Phase::kPortfolioP1
      : MVSState::Phase::kPortfolioP2;

  std::function<void(const State&)> traverse = [&](const State& state) {
    if (state.IsTerminal()) return;

    auto* mvs_state =
        dynamic_cast<const MVSStateWithSubtreePureStrategies*>(&state);
    if (!mvs_state) return;

    // At the opponent's portfolio choice, build full distribution with
    // prob 1 on model entry (last action), 0 on all others
    if (mvs_state->GetPhase() == opponent_phase) {
      std::string info_state = state.InformationStateString(opponent);
      if (result.count(info_state) == 0) {
        auto legal = state.LegalActions();
        SPIEL_CHECK_FALSE(legal.empty());
        // The model entry is appended last, so it's the last legal action
        Action model_action = legal.back();
        ActionsAndProbs dist;
        dist.reserve(legal.size());
        for (Action a : legal) {
          dist.push_back({a, (a == model_action) ? 1.0 : 0.0});
        }
        result[info_state] = std::move(dist);
      }
      return;  // Don't traverse deeper into MVS phases
    }

    // At matrix terminal, stop
    if (mvs_state->GetPhase() == MVSState::Phase::kMatrixTerminal) return;

    // At other portfolio phases (target player's), traverse all actions
    if (mvs_state->GetPhase() != MVSState::Phase::kNormal) {
      for (Action a : state.LegalActions()) {
        auto child = state.Clone();
        child->ApplyAction(a);
        traverse(*child);
      }
      return;
    }

    // Normal phase: traverse game tree
    if (state.IsChanceNode()) {
      for (const auto& [a, p] : state.ChanceOutcomes()) {
        auto child = state.Clone();
        child->ApplyAction(a);
        traverse(*child);
      }
    } else {
      for (Action a : state.LegalActions()) {
        auto child = state.Clone();
        child->ApplyAction(a);
        traverse(*child);
      }
    }
  };

  traverse(*mvs_game.NewInitialState());
  return result;
}

// ============================================================================
// PrecomputeMVSPortfolioDistributions
// ============================================================================

std::unordered_map<std::string, ActionsAndProbs>
PrecomputeMVSPortfolioDistributions(
    const MVSGameWithSubtreePureStrategies& mvs_game,
    const Policy& opponent_model,
    Player opponent) {
  std::unordered_map<std::string, ActionsAndProbs> result;

  // Determine which portfolio phase corresponds to the opponent
  // P0 selects at kPortfolioP1, P1 selects at kPortfolioP2
  auto opponent_phase = (opponent == 0)
      ? MVSState::Phase::kPortfolioP1
      : MVSState::Phase::kPortfolioP2;

  std::function<void(const State&)> traverse = [&](const State& state) {
    if (state.IsTerminal()) return;

    auto* mvs_state =
        dynamic_cast<const MVSStateWithSubtreePureStrategies*>(&state);
    if (!mvs_state) return;

    // At the opponent's portfolio choice, compute model's distribution
    if (mvs_state->GetPhase() == opponent_phase) {
      std::string info_state = state.InformationStateString(opponent);
      // Only compute once per unique info state
      if (result.count(info_state) > 0) return;

      // Get the opponent's portfolio (pure strategies)
      const auto& portfolio = (opponent == 0)
          ? mvs_state->GetPortfolioP0()
          : mvs_state->GetPortfolioP1();

      // Get subtree info states for the opponent
      SubtreeInfostates subtree_is =
          CollectSubtreeInfostates(mvs_state->GetUnderlyingState(), opponent);

      // Compute probability of each pure strategy under the model
      ActionsAndProbs dist;
      for (int k = 0; k < static_cast<int>(portfolio.size()); ++k) {
        double prob = 1.0;
        for (const auto& is : subtree_is.infostates) {
          auto pure_ap = portfolio[k]->GetStatePolicy(is);
          if (pure_ap.empty()) continue;
          // Find the action this pure strategy plays at this info state
          Action chosen = pure_ap[0].first;  // Pure strategy: first (only) entry
          // Get model's probability for this action
          auto model_ap = opponent_model.GetStatePolicy(is);
          prob *= GetProb(model_ap, chosen);
        }
        dist.push_back({k, prob});
      }
      result[info_state] = dist;
      return;  // Don't traverse deeper into MVS phases
    }

    // At matrix terminal, stop
    if (mvs_state->GetPhase() == MVSState::Phase::kMatrixTerminal) return;

    // At other portfolio phase (target player's), traverse all actions
    if (mvs_state->GetPhase() != MVSState::Phase::kNormal) {
      for (Action a : state.LegalActions()) {
        auto child = state.Clone();
        child->ApplyAction(a);
        traverse(*child);
      }
      return;
    }

    // Normal phase: traverse game tree
    if (state.IsChanceNode()) {
      for (const auto& [a, p] : state.ChanceOutcomes()) {
        auto child = state.Clone();
        child->ApplyAction(a);
        traverse(*child);
      }
    } else {
      for (Action a : state.LegalActions()) {
        auto child = state.Clone();
        child->ApplyAction(a);
        traverse(*child);
      }
    }
  };

  traverse(*mvs_game.NewInitialState());
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

      if (config.gadget == GadgetType::kNone) {
        // Unsafe subgame: no T/F choice, extract both players' strategies.
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
        auto subgame_game = CreateUnsafeSubgame(
            decomp.game, std::move(subgame_roots), std::move(total_reaches));
        std::string prefix = "unsafe:subgame:";

        GadgetPolicyWrapper wrapper(config.opponent_model, prefix, -1, 0);
        algorithms::RNRSolver solver(*subgame_game, target, &wrapper, config.p,
                                     true, true, true);
        for (int i = 0; i < config.cfr_iterations; ++i) {
          solver.EvaluateAndUpdatePolicy();
        }

        TabularPolicy policy = solver.TabularAveragePolicy();
        for (const auto& [sub_is, ap] : policy.PolicyTable()) {
          if (sub_is.compare(0, prefix.length(), prefix) == 0) {
            std::string orig = sub_is.substr(prefix.length());
            if (info_per_player[target].count(orig) > 0 ||
                info_per_player[opponent].count(orig) > 0) {
              combined->SetStatePolicy(orig, ap);
            }
          }
        }
      } else {
        // Gadget cases (kResolving or kMaxMargin).
        // The resolving player has artificial T/F (or info-set choice) actions
        // that distort their subgame strategy. So we must use two gadgets:
        //   1. Gadget with res=opponent: solve with RNR → extract target's
        //      strategy (target is non-resolving, strategy is undistorted)
        //   2. Gadget with res=target: solve with CFR → extract opponent's
        //      strategy (opponent is non-resolving, strategy is undistorted)

        // --- Gadget 1: target player's strategy via RNR ---
        {
          int res = opponent;
          auto gadget_roots =
              BuildSubgameRoots(roots, res, decomp.reach_probs[target]);
          if (gadget_roots.empty()) continue;

          std::shared_ptr<const Game> gadget_game;
          std::string prefix;
          int follow_action = -1;
          int num_choice_actions = 0;

          if (config.gadget == GadgetType::kResolving) {
            gadget_game = CreateGadgetGame(
                decomp.game, std::move(gadget_roots), res, decomp.cfvs[res]);
            prefix = "gadget_F:subgame:";
            follow_action = GadgetGame::kFollowAction;
          } else {  // kMaxMargin
            auto mm_game = CreateMaxMarginGadgetGame(
                decomp.game, std::move(gadget_roots), res, decomp.cfvs[res]);
            num_choice_actions = mm_game->NumInfoSets();
            gadget_game = mm_game;
            prefix = "mm_F:subgame:";
          }

          GadgetPolicyWrapper wrapper(config.opponent_model, prefix,
                                      follow_action, num_choice_actions);
          algorithms::RNRSolver solver(*gadget_game, target, &wrapper, config.p,
                                       true, true, true);
          for (int i = 0; i < config.cfr_iterations; ++i) {
            solver.EvaluateAndUpdatePolicy();
          }

          // Extract only target player's strategy (non-resolving).
          TabularPolicy policy = solver.TabularAveragePolicy();
          for (const auto& [sub_is, ap] : policy.PolicyTable()) {
            if (sub_is.compare(0, prefix.length(), prefix) == 0) {
              std::string orig = sub_is.substr(prefix.length());
              if (info_per_player[target].count(orig) > 0) {
                combined->SetStatePolicy(orig, ap);
              }
            }
          }
        }

        // --- Gadget 2: opponent's strategy via CFR ---
        {
          int res = target;
          auto gadget_roots =
              BuildSubgameRoots(roots, res, decomp.reach_probs[opponent]);
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

          algorithms::CFRSolverBase cfr_solver(*gadget_game, true, true, true);
          for (int i = 0; i < config.cfr_iterations; ++i) {
            cfr_solver.EvaluateAndUpdatePolicy();
          }

          // Extract only opponent's strategy (non-resolving).
          TabularPolicy policy = cfr_solver.TabularAveragePolicy();
          for (const auto& [sub_is, ap] : policy.PolicyTable()) {
            if (sub_is.compare(0, prefix.length(), prefix) == 0) {
              std::string orig = sub_is.substr(prefix.length());
              if (info_per_player[opponent].count(orig) > 0) {
                combined->SetStatePolicy(orig, ap);
              }
            }
          }
        }
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
  // Tabularize opponent model so string-based GetStatePolicy works.
  // Use shared_ptr so it can be passed to the MVS game as the model entry.
  auto opponent_tabular = std::make_shared<TabularPolicy>(
      TabularizePolicy(*game, opponent_model));

  int opponent = 1 - config.target_player;

  // Step 1: Create and solve MVS trunk.
  // Pass the opponent model as an extra portfolio entry for the opponent player.
  auto mvs_game = CreateMVSGameWithSubtreePureStrategies(
      game, depth, depth_mode, opponent_tabular, opponent);

  auto trunk_is = (depth_mode == MVSGame::DepthMode::kRoundBased)
      ? CollectInfoStateStringsBeforeRound(*game, depth)
      : CollectInfoStateStringsBeforeDepth(*game, depth);

  // Use RNR for the trunk to respect the p parameter.
  // MVSModelEntryPolicy handles both regular info states (delegating to the
  // opponent model) and MVS portfolio choice info states for the opponent
  // (always selecting the model entry, i.e., the last action).
  auto model_indices = PrecomputeModelActionIndices(*mvs_game, opponent);
  MVSModelEntryPolicy mvs_opponent(opponent_tabular.get(),
                                   std::move(model_indices));

  algorithms::RNRSolver trunk_solver(
      *mvs_game, config.target_player, &mvs_opponent, config.p,
      true, true, true);

  for (int i = 0; i < config.cfr_iterations; ++i) {
    trunk_solver.EvaluateAndUpdatePolicy();
  }
  auto mvs_policy = trunk_solver.AveragePolicy();

  // Step 2: Extract trunk policy into original game's info state space
  // Both players' strategies come from the RNR solution.
  // At p=0 this is Nash; at p=1 target best-responds, opponent best-responds
  // to target; at intermediate p both are the RNR equilibrium strategies.
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
  subgame_config.opponent_model = opponent_tabular.get();

  return ResolveSubgames(decomp, trunk, subgame_config);
}

}  // namespace open_spiel
