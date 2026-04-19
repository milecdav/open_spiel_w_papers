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

#include "open_spiel/algorithms/best_response.h"
#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/algorithms/rnr.h"
#if OPEN_SPIEL_BUILD_WITH_ORTOOLS
#include "open_spiel/algorithms/ortools/sequence_form_lp.h"
#endif
#include "open_spiel/game_transforms/max_margin_gadget.h"
#include "open_spiel/game_transforms/matrix_valued_states.h"
#include "open_spiel/game_transforms/resolving_gadget.h"
#include "open_spiel/game_transforms/ox_gadget.h"
#include "open_spiel/game_transforms/ses_gadget.h"
#include "open_spiel/game_transforms/full_gadget.h"
#include "open_spiel/game_transforms/gadget.h"
#include "open_spiel/game_transforms/subgame_utils.h"
#include "open_spiel/game_transforms/rnr_mixture.h"
#include "open_spiel/game_transforms/unsafe_subgame.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"

namespace open_spiel {

namespace {

// Tabularize a policy by traversing the game tree.
// Uses the State-based GetStatePolicy (works for UniformPolicy etc.)
TabularPolicy TabularizePolicy(const Game& game, const Policy& policy) {
  TabularPolicy result;
  int num_players = game.NumPlayers();
  std::function<void(const State&)> traverse = [&](const State& state) {
    if (state.IsTerminal()) return;
    if (state.IsChanceNode()) {
      for (const auto& [a, p] : state.ChanceOutcomes()) {
        auto child = state.Clone();
        child->ApplyAction(a);
        traverse(*child);
      }
    } else if (state.IsSimultaneousNode()) {
      // Simultaneous game: store per-player policies for each player.
      for (Player pl = 0; pl < num_players; ++pl) {
        auto ap = policy.GetStatePolicy(state, pl);
        if (!ap.empty()) {
          result.SetStatePolicy(state.InformationStateString(pl), ap);
        }
      }
      for (Action a : state.LegalActions()) {
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

// Solve a transformed game using either CFR or LP and return the policy.
// For LP: uses sequence-form LP (exact). For CFR: uses alternating CFR.
TabularPolicy SolveTransformedGame(const Game& game, SolverType solver_type,
                                   int cfr_iterations) {
  if (solver_type == SolverType::kLP) {
#if OPEN_SPIEL_BUILD_WITH_ORTOOLS
    auto [policy, value] =
        algorithms::ortools::MakeEquilibriumPolicy(game, true);
    return policy;
#else
    SpielFatalError("LP solver requested but OPEN_SPIEL_BUILD_WITH_ORTOOLS "
                    "is not enabled. Build with OPEN_SPIEL_BUILD_WITH_ORTOOLS=ON.");
#endif
  }
  // Default: CFR
  algorithms::CFRSolverBase solver(game, true, true, true);
  for (int i = 0; i < cfr_iterations; ++i) {
    solver.EvaluateAndUpdatePolicy();
  }
  return solver.TabularAveragePolicy();
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
// Helper: compute correct joint reach for unsafe subgames
// ============================================================================

namespace {

// Compute joint reach = r0 * r1 / chance_reach when chance_reach is available,
// otherwise fall back to r0 * r1 (which double-counts chance).
// This is needed because reach_probs[i] = chance * player_i_actions,
// so r0 * r1 = chance^2 * p0 * p1, but we want chance * p0 * p1.
double ComputeJointReach(
    const std::string& hist,
    const SubgameDecomposition& decomp) {
  double r0 = 0, r1 = 0;
  auto it0 = decomp.reach_probs[0].find(hist);
  if (it0 != decomp.reach_probs[0].end()) r0 = it0->second;
  auto it1 = decomp.reach_probs[1].find(hist);
  if (it1 != decomp.reach_probs[1].end()) r1 = it1->second;

  double total = r0 * r1;
  // Correct for chance double-counting if chance_reach is available
  auto it_c = decomp.chance_reach.find(hist);
  if (it_c != decomp.chance_reach.end() && it_c->second > 0) {
    total /= it_c->second;
  }
  return total;
}

// Compute expected returns at a state under a policy (recursive)
std::vector<double> ComputeExpectedReturnsUnderPolicy(
    const State& state, const Policy& policy) {
  if (state.IsTerminal()) return state.Returns();
  int np = state.NumPlayers();
  std::vector<double> ev(np, 0.0);
  if (state.IsChanceNode()) {
    for (const auto& [a, p] : state.ChanceOutcomes()) {
      auto child = state.Clone();
      child->ApplyAction(a);
      auto cev = ComputeExpectedReturnsUnderPolicy(*child, policy);
      for (int i = 0; i < np; ++i) ev[i] += p * cev[i];
    }
  } else {
    Player pl = state.CurrentPlayer();
    auto ap = policy.GetStatePolicy(state, pl);
    if (ap.empty()) {
      SpielFatalError("Missing policy entry in ComputeExpectedReturnsUnderPolicy");
    }
    for (const auto& [a, p] : ap) {
      if (p <= 0.0) continue;
      auto child = state.Clone();
      child->ApplyAction(a);
      auto cev = ComputeExpectedReturnsUnderPolicy(*child, policy);
      for (int i = 0; i < np; ++i) ev[i] += p * cev[i];
    }
  }
  return ev;
}

GadgetContext BuildGadgetContext(
    const SubgameDecomposition& decomp, const std::string& pub_obs,
    const std::vector<std::unique_ptr<State>>& roots, Player resolving_player,
    const ResolvingConfig& config) {
  GadgetContext ctx;
  ctx.original_game = decomp.game;
  ctx.roots = &roots;
  ctx.pub_obs = pub_obs;
  ctx.cfvs = {&decomp.cfvs[0], &decomp.cfvs[1]};
  ctx.reach_probs = {&decomp.reach_probs[0], &decomp.reach_probs[1]};
  ctx.chance_reach = &decomp.chance_reach;
  ctx.resolving_player = resolving_player;
  ctx.opponent_model = config.opponent_model;
  return ctx;
}

void ExtractResolvingStrategyInto(
    const TabularPolicy& policy, const std::string& is_prefix, Player player,
    const std::vector<std::unique_ptr<State>>& roots, TabularPolicy* out) {
  (void)roots;
  SPIEL_CHECK_TRUE(out != nullptr);
  for (const auto& [sub_is, ap] : policy.PolicyTable()) {
    if (is_prefix.empty()) {
      // Empty prefix means entries are already in original-game key space.
      // Skip known synthetic prefixes from transformed wrappers.
      if (sub_is.find("rr_") == 0 || sub_is == "rnr_root" ||
          sub_is == "unsafe_start") {
        continue;
      }
      out->SetStatePolicy(sub_is, ap);
      continue;
    }

    if (sub_is.compare(0, is_prefix.size(), is_prefix) != 0) continue;
    std::string orig_is = sub_is.substr(is_prefix.size());

    // Full gadget may emit player-tagged keys, e.g. "P0:<orig_is>".
    if (orig_is.size() > 3 && orig_is[0] == 'P' && orig_is[2] == ':') {
      int tagged_player = orig_is[1] - '0';
      if (tagged_player != player) continue;
      orig_is = orig_is.substr(3);
    }
    out->SetStatePolicy(orig_is, ap);
  }
}

TabularPolicy SolveNashGame(const Game& game, SolverType solver_type,
                            int cfr_iterations) {
  return SolveTransformedGame(game, solver_type, cfr_iterations);
}

std::shared_ptr<const UnsafeSubgameGame> BuildUnsafeSubgameWithOpponentModelReach(
    const GadgetContext& ctx, Player opponent) {
  SPIEL_CHECK_TRUE(ctx.original_game != nullptr);
  SPIEL_CHECK_TRUE(ctx.roots != nullptr);
  SPIEL_CHECK_TRUE(ctx.opponent_model != nullptr);

  std::vector<const State*> root_ptrs;
  root_ptrs.reserve(ctx.roots->size());
  for (const auto& root : *ctx.roots) root_ptrs.push_back(root.get());
  auto opp_reach = ComputeReachProbabilities(*ctx.original_game, *ctx.opponent_model,
                                             opponent, root_ptrs);

  std::vector<std::unique_ptr<State>> subgame_roots;
  std::vector<double> reaches;
  for (const auto& root : *ctx.roots) {
    const std::string hist = root->HistoryString();
    auto it = opp_reach.find(hist);
    if (it != opp_reach.end() && it->second > 0.0) {
      subgame_roots.push_back(root->Clone());
      reaches.push_back(it->second);
    }
  }
  if (subgame_roots.empty()) return nullptr;
  return CreateUnsafeSubgame(ctx.original_game, std::move(subgame_roots),
                             std::move(reaches));
}

std::function<std::string(const State&, Player)> MakeCanonicalizerForSubgames(
    const std::string& free_prefix, const std::string& fixed_prefix) {
  return MakeSubgameISCanonicalizer(free_prefix, fixed_prefix);
}

struct EffectiveResolvingConfig {
  SolverType legacy_solver;
  GadgetType legacy_gadget;
  SolverKind solver_kind;
  ResponseKind response_kind;
  GadgetKind gadget_kind;
};

EffectiveResolvingConfig ResolveEffectiveConfig(const ResolvingConfig& config) {
  EffectiveResolvingConfig eff;
  eff.solver_kind = config.solver_kind;
  eff.response_kind = config.response_kind;
  eff.gadget_kind = config.gadget_kind;

  // Backward-compatible overrides from legacy fields.
  // If caller explicitly set solver_kind=kLP on the new API, preserve it.
  const bool explicit_lp = (config.solver_kind == SolverKind::kLP);
  if (config.solver == SolverType::kRNR) {
    eff.response_kind = ResponseKind::kRNR;
    if (!explicit_lp) eff.solver_kind = SolverKind::kCFR;
  } else if (config.solver == SolverType::kLP) {
    eff.solver_kind = SolverKind::kLP;
  } else {
    if (!explicit_lp) eff.solver_kind = SolverKind::kCFR;
  }

  switch (config.gadget) {
    case GadgetType::kNone:
      eff.gadget_kind = GadgetKind::kUnsafe;
      break;
    case GadgetType::kResolving:
      eff.gadget_kind = GadgetKind::kResolving;
      break;
    case GadgetType::kMaxMargin:
      eff.gadget_kind = GadgetKind::kMaxMargin;
      break;
    case GadgetType::kFullPath:
      eff.gadget_kind = GadgetKind::kFullPath;
      break;
    case GadgetType::kFullTrunk:
      eff.gadget_kind = GadgetKind::kFullTrunk;
      break;
    case GadgetType::kSES:
    case GadgetType::kOX:
      // Legacy-only gadgets kept during transition.
      break;
  }

  // Reconstruct legacy dispatch target from new axes for current code path.
  eff.legacy_solver = (eff.response_kind == ResponseKind::kRNR)
                          ? SolverType::kRNR
                          : (eff.solver_kind == SolverKind::kLP
                                 ? SolverType::kLP
                                 : SolverType::kCFR);

  if (config.gadget == GadgetType::kSES || config.gadget == GadgetType::kOX) {
    eff.legacy_gadget = config.gadget;
  } else {
    switch (eff.gadget_kind) {
      case GadgetKind::kUnsafe:
        eff.legacy_gadget = GadgetType::kNone;
        break;
      case GadgetKind::kResolving:
        eff.legacy_gadget = GadgetType::kResolving;
        break;
      case GadgetKind::kMaxMargin:
        eff.legacy_gadget = GadgetType::kMaxMargin;
        break;
      case GadgetKind::kFullPath:
        eff.legacy_gadget = GadgetType::kFullPath;
        break;
      case GadgetKind::kFullTrunk:
        eff.legacy_gadget = GadgetType::kFullTrunk;
        break;
      case GadgetKind::kResolvingByIS:
        eff.legacy_gadget = GadgetType::kResolvingByIS;
        break;
    }
  }
  return eff;
}

}  // namespace

// ============================================================================
// ResolveSubgames
// ============================================================================

std::shared_ptr<TabularPolicy> ResolveSubgames(
    const SubgameDecomposition& decomp,
    const TabularPolicy& trunk_policy,
    const ResolvingConfig& config) {
  auto combined = std::make_shared<TabularPolicy>();
  const EffectiveResolvingConfig eff = ResolveEffectiveConfig(config);

  // Copy trunk
  for (const auto& [player, info_states] : decomp.trunk_info_states) {
    for (const auto& is : info_states) {
      auto ap = trunk_policy.GetStatePolicy(is);
      if (!ap.empty()) combined->SetStatePolicy(is, ap);
    }
  }

  if (eff.legacy_solver == SolverType::kCFR ||
      eff.legacy_solver == SolverType::kLP) {
    // Standard equilibrium solving (CFR or LP)
    if (eff.legacy_gadget == GadgetType::kNone) {
      // Unsafe: solve once, extract both players' strategies
      for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
        std::vector<std::unique_ptr<State>> subgame_roots;
        std::vector<double> total_reaches;
        for (const auto& root : roots) {
          std::string hist = root->HistoryString();
          double total = ComputeJointReach(hist, decomp);
          if (total > 0) {
            subgame_roots.push_back(root->Clone());
            total_reaches.push_back(total);
          }
        }
        if (subgame_roots.empty()) continue;

        auto unsafe = CreateUnsafeSubgame(
            decomp.game, std::move(subgame_roots), std::move(total_reaches));
        TabularPolicy policy = SolveTransformedGame(
            *unsafe, eff.legacy_solver, config.cfr_iterations);
        const std::string prefix = "unsafe:subgame:";
        for (const auto& [sub_is, ap] : policy.PolicyTable()) {
          if (sub_is.compare(0, prefix.length(), prefix) == 0) {
            combined->SetStatePolicy(sub_is.substr(prefix.length()), ap);
          }
        }
      }
    } else if (eff.legacy_gadget == GadgetType::kSES) {
      // SES gadget: solve per-player using SES gadget
      // SES requires an opponent model for computing model reach p̂(I)
      SPIEL_CHECK_TRUE(config.opponent_model != nullptr);
      for (int non_res = 0; non_res < 2; ++non_res) {
        int res = 1 - non_res;
        for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
          auto info_per_player = CollectSubgameInfoStatesPerPlayer(roots);

          // Build SES roots: info_state_string = NON-resolving player's IS,
          // reach_prob = resolving player's reach (π_{-nonres})
          auto ses_roots =
              BuildSubgameRoots(roots, non_res, decomp.reach_probs[res]);
          if (ses_roots.empty()) continue;

          // Compute model info set reach: aggregate non-resolving player's
          // reach under the model, grouped by non-resolving info state
          std::vector<const State*> root_ptrs;
          for (const auto& root : roots) root_ptrs.push_back(root.get());
          auto model_reach = ComputeReachProbabilities(
              *decomp.game, *config.opponent_model, non_res, root_ptrs);

          std::unordered_map<std::string, double> model_info_set_reach;
          for (const auto& root : roots) {
            std::string non_res_is = root->InformationStateString(non_res);
            auto it = model_reach.find(root->HistoryString());
            if (it != model_reach.end()) {
              model_info_set_reach[non_res_is] += it->second;
            }
          }

          // CFVs for the non-resolving player (keyed by non-resolving IS)
          auto ses_game = CreateSESGadgetGame(
              decomp.game, std::move(ses_roots), res, decomp.cfvs[non_res],
              config.alpha, model_info_set_reach);

          TabularPolicy policy = SolveTransformedGame(
              *ses_game, eff.legacy_solver, config.cfr_iterations);

          // In SES, extract the RESOLVING player's strategy (they only act
          // in the subgame). The non-resolving player has artificial info set
          // choice actions that distort their strategy.
          const std::string prefix = "ses_F:subgame:";
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
    } else if (eff.legacy_gadget == GadgetType::kOX) {
      // OX gadget: solve per-player using OX gadget
      // OX requires an opponent model for computing model reach p̂(I) and CBV
      SPIEL_CHECK_TRUE(config.opponent_model != nullptr);
      for (int non_res = 0; non_res < 2; ++non_res) {
        int res = 1 - non_res;

        // Compute CBV for the non-resolving player using TabularBestResponse
        algorithms::TabularBestResponse br(*decomp.game, non_res, &trunk_policy);

        for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
          auto info_per_player = CollectSubgameInfoStatesPerPlayer(roots);

          // Build OX roots: info_state_string = NON-resolving player's IS,
          // reach_prob = resolving player's reach (π_{-nonres})
          auto ox_roots =
              BuildSubgameRoots(roots, non_res, decomp.reach_probs[res]);
          if (ox_roots.empty()) continue;

          // Compute CBV: CBV(I) = Σ_{h∈I} π_{-nonres}(h) × br.Value(h)
          std::unordered_map<std::string, double> cbv_values;
          for (const auto& root : roots) {
            std::string info_state = root->InformationStateString(non_res);
            std::string hist = root->HistoryString();
            auto reach_it = decomp.reach_probs[res].find(hist);
            double reach = (reach_it != decomp.reach_probs[res].end())
                               ? reach_it->second
                               : 0.0;
            if (reach > 0) {
              cbv_values[info_state] += reach * br.Value(hist);
            }
          }

          // Compute model info set reach
          std::vector<const State*> root_ptrs;
          for (const auto& root : roots) root_ptrs.push_back(root.get());
          auto model_reach = ComputeReachProbabilities(
              *decomp.game, *config.opponent_model, non_res, root_ptrs);

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
              config.beta, model_info_set_reach);

          TabularPolicy policy = SolveTransformedGame(
              *ox_game, eff.legacy_solver, config.cfr_iterations);

          // Extract the RESOLVING player's strategy
          const std::string prefix = "ox_F:subgame:";
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
    } else if (eff.legacy_gadget == GadgetType::kFullPath ||
               eff.legacy_gadget == GadgetType::kFullTrunk) {
      // Full Gadget: keeps actual trunk game structure for exact exploitability
      FullGadgetGame::Mode mode = (eff.legacy_gadget == GadgetType::kFullPath)
                                      ? FullGadgetGame::Mode::kPath
                                      : FullGadgetGame::Mode::kTrunk;

      // Precompute boundary grouping for Full Gadget
      std::unordered_map<std::string, std::vector<std::string>>
          boundary_by_group;

      for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
        for (const auto& root : roots) {
          std::string hist = root->HistoryString();
          boundary_by_group[pub_obs].push_back(hist);
        }
      }

      auto trunk_policy_ptr = std::make_shared<TabularPolicy>(trunk_policy);

      // Solve per-player (as resolving player)
      int num_groups = decomp.grouped_subgames.size();
      int group_idx = 0;
      for (int res = 0; res < 2; ++res) {
        group_idx = 0;
        for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
          ++group_idx;
          auto info_per_player = CollectSubgameInfoStatesPerPlayer(roots);

          // Use per-state lazy portfolio enumeration at non-target boundaries
          auto fg = CreateFullGadgetGame(
              decomp.game, trunk_policy_ptr, res, pub_obs,
              boundary_by_group, mode,
              /*boundary_portfolios_p0=*/{},
              /*boundary_portfolios_p1=*/{},
              /*enumerate_boundary_portfolios=*/true);

          std::cerr << "  [Full Gadget] res=" << res
                    << " group " << group_idx << "/" << num_groups
                    << " (" << pub_obs << ", " << roots.size() << " roots)"
                    << std::endl;

          TabularPolicy policy = SolveTransformedGame(
              *fg, eff.legacy_solver, config.cfr_iterations);

          // Extract resolving player's strategy from subgame info states
          const std::string prefix = "full_F:subgame:";
          for (const auto& [sub_is, ap] : policy.PolicyTable()) {
            if (sub_is.compare(0, prefix.length(), prefix) == 0) {
              std::string orig = sub_is.substr(prefix.length());
              // Support keys with explicit player tag: full_F:subgame:P<id>:<orig>
              if (orig.size() > 3 && orig[0] == 'P' &&
                  (orig[1] == '0' || orig[1] == '1') && orig[2] == ':') {
                orig = orig.substr(3);
              }
              if (info_per_player[res].count(orig) > 0) {
                combined->SetStatePolicy(orig, ap);
              }
            }
          }
        }
      }
    } else {
      // With gadget (kResolving or kMaxMargin): solve per-player
      for (int res = 0; res < 2; ++res) {
        int non_res = 1 - res;
        for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
          auto info_per_player = CollectSubgameInfoStatesPerPlayer(roots);
          // Roots grouped by non-resolving (adversary) player's info states,
          // with resolving player's reach probabilities
          auto gadget_roots =
              BuildSubgameRoots(roots, non_res, decomp.reach_probs[res]);
          if (gadget_roots.empty()) continue;

          std::shared_ptr<const Game> gadget_game;
          std::string prefix;

          // adversary_player = non_res (has artificial actions)
          if (eff.legacy_gadget == GadgetType::kResolving) {
            gadget_game = CreateGadgetGame(
                decomp.game, std::move(gadget_roots), non_res,
                decomp.cfvs[non_res]);
            prefix = "gadget_F:subgame:";
          } else {  // kMaxMargin
            gadget_game = CreateMaxMarginGadgetGame(
                decomp.game, std::move(gadget_roots), non_res,
                decomp.cfvs[non_res]);
            prefix = "mm_F:subgame:";
          }

          TabularPolicy policy = SolveTransformedGame(
              *gadget_game, eff.legacy_solver, config.cfr_iterations);

          // Extract the resolving player's strategy
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
    }
  } else {
    // Canonical RNR path: solve an explicit mixture game on target pass, and
    // solve plain gadget game for opponent pass when gadget distorts opponent.
    SPIEL_CHECK_TRUE(config.opponent_model != nullptr);
    const int target = config.target_player;
    const int opponent = 1 - target;
    const SolverType nash_solver =
        (eff.solver_kind == SolverKind::kLP) ? SolverType::kLP : SolverType::kCFR;

    // Special case: kNone (unsafe) with RNR. The two-pass approach (separate
    // opponent and target passes) fails at p=1 because the target's best-
    // response assigns zero reach to some subgame roots, making joint reach
    // zero and causing Build() to return null for those subgames. Use the
    // original single-pass approach: build one joint unsafe subgame, run
    // RNRSolver directly, and extract both players' strategies.
    if (eff.legacy_gadget == GadgetType::kNone) {
      for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
        auto info_per_player = CollectSubgameInfoStatesPerPlayer(roots);

        std::vector<std::unique_ptr<State>> subgame_roots;
        std::vector<double> total_reaches;
        for (const auto& root : roots) {
          std::string hist = root->HistoryString();
          double total = ComputeJointReach(hist, decomp);
          if (total > 0) {
            subgame_roots.push_back(root->Clone());
            total_reaches.push_back(total);
          }
        }
        if (subgame_roots.empty()) continue;

        auto subgame_game = CreateUnsafeSubgame(
            decomp.game, std::move(subgame_roots), std::move(total_reaches));
        const std::string prefix = "unsafe:subgame:";

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
      }
      return combined;
    }

    std::unordered_map<std::string, std::vector<std::string>> boundary_by_group;
    for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
      for (const auto& root : roots) {
        boundary_by_group[pub_obs].push_back(root->HistoryString());
      }
    }
    auto trunk_policy_ptr = std::make_shared<TabularPolicy>(trunk_policy);

    std::unique_ptr<Gadget> gadget;
    if (eff.legacy_gadget == GadgetType::kResolving) {
      gadget = MakeResolvingGadget();
    } else if (eff.legacy_gadget == GadgetType::kResolvingByIS) {
      gadget = MakeResolvingByISGadget();
    } else if (eff.legacy_gadget == GadgetType::kMaxMargin) {
      gadget = MakeMaxMarginGadget();
    } else if (eff.legacy_gadget == GadgetType::kFullPath) {
      gadget = MakeFullGadget(FullGadgetGame::Mode::kPath, trunk_policy_ptr,
                              boundary_by_group);
    } else if (eff.legacy_gadget == GadgetType::kFullTrunk) {
      gadget = MakeFullGadget(FullGadgetGame::Mode::kTrunk, trunk_policy_ptr,
                              boundary_by_group);
    } else {
      SpielFatalError("RNR mixture dispatcher: unsupported legacy gadget.");
    }
    SPIEL_CHECK_TRUE(gadget != nullptr);

    // Run opponent pass first so target pass (mixture) writes final target rows.
    // Even for non-distorting gadgets, this ensures opponent rows are filled in
    // original-game key space from the plain gadget game.
    std::vector<Player> passes = {opponent, target};

    for (Player resolving : passes) {
      const bool rnr_pass = (resolving == target);
      for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
        GadgetContext ctx = BuildGadgetContext(
            decomp, pub_obs, roots, resolving, config);
        auto free_game = gadget->Build(ctx);
        if (!free_game) continue;

        std::shared_ptr<const Game> game_to_solve;
        std::string extract_prefix;

        if (rnr_pass) {
          auto fixed_game =
              BuildUnsafeSubgameWithOpponentModelReach(ctx, opponent);
          if (!fixed_game) continue;
          auto canonicalizer = MakeCanonicalizerForSubgames(
              gadget->SubgamePrefix(), "unsafe:subgame:");
          double fixed_multiplier = 1.0;
          if (const auto* rg = dynamic_cast<const GadgetGame*>(free_game.get())) {
            fixed_multiplier = rg->NormalizationConstant();
          }
          game_to_solve = CreateRNRMixtureGame(
              free_game, fixed_game, target,
              std::shared_ptr<const Policy>(config.opponent_model,
                                            [](const Policy*) {}),
              config.p, config.lock_opponent_in_fixed_branch,
              fixed_multiplier, canonicalizer);
          extract_prefix = "";
        } else {
          game_to_solve = free_game;
          extract_prefix = gadget->SubgamePrefix();
        }

        TabularPolicy solved =
            SolveNashGame(*game_to_solve, nash_solver, config.cfr_iterations);
        ExtractResolvingStrategyInto(
            solved, extract_prefix, resolving, roots, combined.get());
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
  const EffectiveResolvingConfig eff = ResolveEffectiveConfig(config);
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
  auto mvs_opponent_shared = std::make_shared<MVSModelEntryPolicy>(
      opponent_tabular.get(), std::move(model_indices));

  std::shared_ptr<Policy> mvs_policy;

  if (eff.solver_kind == SolverKind::kLP) {
#if OPEN_SPIEL_BUILD_WITH_ORTOOLS
    // Wrap MVS in an RNR mixture game so LP can compute the RNR equilibrium.
    // Free branch: opponent is a free player in MVS.
    // Fixed branch (lock=true): opponent is chance drawn from
    // MVSModelEntryPolicy (picks model entry at portfolio nodes, delegates
    // elsewhere). Target IS are shared across branches via the identity
    // canonicalizer (MVS IS strings are their own canonical form).
    auto identity_canon =
        [](const State& s, Player p) -> std::string {
      return s.InformationStateString(p);
    };
    auto mvs_mixture = CreateRNRMixtureGame(
        mvs_game, mvs_game, config.target_player,
        std::static_pointer_cast<const Policy>(mvs_opponent_shared),
        config.p,
        /*lock_opponent_in_fixed_branch=*/true,
        /*fixed_branch_utility_multiplier=*/1.0,
        identity_canon);
    auto mix_pair = algorithms::ortools::MakeEquilibriumPolicy(
        *mvs_mixture, /*uniform_imputation=*/true);
    const TabularPolicy& mix_policy = mix_pair.first;

    // Build an MVS-IS-keyed policy by walking MVS and mixture in parallel
    // along the free branch (at p=1 free branch has zero reach, so walk
    // fixed branch instead; target IS is shared either way).
    auto mvs_tabular = std::make_shared<TabularPolicy>();
    std::function<void(const State&, const State&)> walk =
        [&](const State& mvs_state, const State& mix_state) {
      if (mvs_state.IsTerminal()) return;
      if (mvs_state.IsChanceNode()) {
        for (const auto& [a, prob] : mvs_state.ChanceOutcomes()) {
          auto ms = mvs_state.Clone(); ms->ApplyAction(a);
          auto xs = mix_state.Clone(); xs->ApplyAction(a);
          walk(*ms, *xs);
        }
        return;
      }
      Player pl = mvs_state.CurrentPlayer();
      std::string mix_is = mix_state.InformationStateString(pl);
      auto ap = mix_policy.GetStatePolicy(mix_is);
      if (!ap.empty()) {
        mvs_tabular->SetStatePolicy(
            mvs_state.InformationStateString(pl), ap);
      }
      for (Action a : mvs_state.LegalActions()) {
        auto ms = mvs_state.Clone(); ms->ApplyAction(a);
        auto xs = mix_state.Clone(); xs->ApplyAction(a);
        walk(*ms, *xs);
      }
    };
    auto mix_root = mvs_mixture->NewInitialState();
    if (config.p < 1.0) {
      mix_root->ApplyAction(RNRMixtureGame::kFreeBranchAction);
    } else {
      mix_root->ApplyAction(RNRMixtureGame::kFixedBranchAction);
    }
    walk(*mvs_game->NewInitialState(), *mix_root);
    mvs_policy = mvs_tabular;
#else
    SpielFatalError(
        "LP trunk requested but OPEN_SPIEL_BUILD_WITH_ORTOOLS is not enabled.");
#endif
  } else {
    algorithms::RNRSolver trunk_solver(
        *mvs_game, config.target_player, mvs_opponent_shared.get(),
        config.p, true, true, true);
    for (int i = 0; i < config.cfr_iterations; ++i) {
      trunk_solver.EvaluateAndUpdatePolicy();
    }
    // Use TabularAveragePolicy (not AveragePolicy) so that copy_trunk and
    // ExtractReachProbsFromMVS can safely call GetStatePolicy without
    // throwing for states not visited during training (e.g. non-model-entry
    // portfolio branches that have zero fixed_opponent_reach at p>0).
    mvs_policy = std::make_shared<TabularPolicy>(
        trunk_solver.TabularAveragePolicy());
  }

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
  decomp.chance_reach = ExtractChanceReachFromMVS(*mvs_game);

  // Fix opponent's reach: the MVS policy gives the *free* opponent's reach,
  // but the actual opponent is (1-p)*free + p*model. At p=1, the free part
  // has weight 0 so its reach is arbitrary. We need to blend with the model's
  // actual reach on the original game.
  if (config.p > 0 && eff.response_kind == ResponseKind::kRNR) {
    // Collect all subgame root states for reach computation
    std::vector<const State*> root_ptrs;
    for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
      for (const auto& root : roots) {
        root_ptrs.push_back(root.get());
      }
    }
    auto model_reach = ComputeReachProbabilities(
        *game, *opponent_tabular, opponent, root_ptrs);

    auto& opp_reach = decomp.reach_probs[opponent];
    // Blend: (1-p) * mvs_reach + p * model_reach
    for (auto& [hist, reach] : opp_reach) {
      auto it = model_reach.find(hist);
      double model_r = (it != model_reach.end()) ? it->second : 0.0;
      reach = (1 - config.p) * reach + config.p * model_r;
    }
    // Add entries from model_reach not in mvs_reach
    for (const auto& [hist, model_r] : model_reach) {
      if (opp_reach.count(hist) == 0) {
        opp_reach[hist] = config.p * model_r;
      }
    }
  }
  decomp.cfvs[0] = ExtractCFVsFromMVSSolution(*mvs_game, *mvs_policy, 0);
  decomp.cfvs[1] = ExtractCFVsFromMVSSolution(*mvs_game, *mvs_policy, 1);

  // Step 4: Resolve subgames
  ResolvingConfig subgame_config = config;
  subgame_config.opponent_model = opponent_tabular.get();

  return ResolveSubgames(decomp, trunk, subgame_config);
}

}  // namespace open_spiel
