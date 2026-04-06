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

#include "open_spiel/game_transforms/subgame_utils.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "open_spiel/algorithms/expected_returns.h"
#include "open_spiel/spiel.h"

namespace open_spiel {

namespace {

void CollectStatesRecursive(
    const State& state,
    const std::function<bool(const State&)>& predicate,
    std::vector<std::unique_ptr<State>>& result) {
  if (state.IsTerminal()) return;

  if (predicate(state)) {
    result.push_back(state.Clone());
    return;  // Frontier: don't recurse past collected states
  }

  if (state.IsChanceNode()) {
    for (const auto& [action, prob] : state.ChanceOutcomes()) {
      auto child = state.Clone();
      child->ApplyAction(action);
      CollectStatesRecursive(*child, predicate, result);
    }
  } else {
    for (Action action : state.LegalActions()) {
      auto child = state.Clone();
      child->ApplyAction(action);
      CollectStatesRecursive(*child, predicate, result);
    }
  }
}

void CollectStatesAtDepthRecursive(
    const State& state,
    int current_depth,
    int target_depth,
    std::vector<std::unique_ptr<State>>& result) {
  if (current_depth == target_depth) {
    result.push_back(state.Clone());
    return;
  }

  if (state.IsTerminal()) return;

  if (state.IsChanceNode()) {
    for (const auto& [action, prob] : state.ChanceOutcomes()) {
      auto child = state.Clone();
      child->ApplyAction(action);
      CollectStatesAtDepthRecursive(*child, current_depth, target_depth,
                                    result);
    }
  } else {
    for (Action action : state.LegalActions()) {
      auto child = state.Clone();
      child->ApplyAction(action);
      CollectStatesAtDepthRecursive(*child, current_depth + 1, target_depth,
                                    result);
    }
  }
}

void CollectStatesAtRoundRecursive(
    const State& state,
    int current_round,
    int target_round,
    std::vector<std::unique_ptr<State>>& result) {
  if (state.IsTerminal()) return;

  if (state.IsChanceNode()) {
    for (const auto& [action, prob] : state.ChanceOutcomes()) {
      auto child = state.Clone();
      child->ApplyAction(action);
      CollectStatesAtRoundRecursive(*child, current_round + 1, target_round,
                                    result);
    }
  } else {
    if (current_round >= target_round) {
      result.push_back(state.Clone());
      return;  // Frontier: don't recurse further
    }
    for (Action action : state.LegalActions()) {
      auto child = state.Clone();
      child->ApplyAction(action);
      CollectStatesAtRoundRecursive(*child, current_round, target_round,
                                    result);
    }
  }
}

void ComputeReachProbsRecursive(
    const State& state,
    const Policy& opponent_policy,
    Player opponent,
    double reach_prob,
    const std::unordered_set<std::string>& target_histories,
    std::unordered_map<std::string, double>& result) {
  std::string hist = state.HistoryString();

  if (target_histories.count(hist) > 0) {
    result[hist] = reach_prob;
    return;
  }

  if (state.IsTerminal()) return;

  if (state.IsChanceNode()) {
    for (auto& [action, prob] : state.ChanceOutcomes()) {
      auto child = state.Clone();
      child->ApplyAction(action);
      ComputeReachProbsRecursive(*child, opponent_policy, opponent,
                                  reach_prob * prob, target_histories, result);
    }
  } else {
    Player player = state.CurrentPlayer();
    auto actions_probs = opponent_policy.GetStatePolicy(state, player);

    for (auto& [action, prob] : actions_probs) {
      auto child = state.Clone();
      child->ApplyAction(action);

      double new_reach = reach_prob;
      if (player == opponent) {
        new_reach *= prob;
      }

      ComputeReachProbsRecursive(*child, opponent_policy, opponent,
                                  new_reach, target_histories, result);
    }
  }
}

void CollectInfoStateStringsBeforeDepthRecursive(
    const State& state,
    int current_depth,
    int depth_limit,
    std::unordered_map<int, std::unordered_set<std::string>>& result) {
  if (state.IsTerminal()) return;

  if (state.IsChanceNode()) {
    for (const auto& [action, prob] : state.ChanceOutcomes()) {
      auto child = state.Clone();
      child->ApplyAction(action);
      CollectInfoStateStringsBeforeDepthRecursive(*child, current_depth, depth_limit,
                                      result);
    }
  } else {
    if (current_depth < depth_limit) {
      Player player = state.CurrentPlayer();
      result[player].insert(state.InformationStateString(player));
    }
    for (Action action : state.LegalActions()) {
      auto child = state.Clone();
      child->ApplyAction(action);
      CollectInfoStateStringsBeforeDepthRecursive(*child, current_depth + 1, depth_limit,
                                      result);
    }
  }
}

void CollectInfoStateStringsBeforeRoundRecursive(
    const State& state,
    int current_round,
    int target_round,
    std::unordered_map<int, std::unordered_set<std::string>>& result) {
  if (state.IsTerminal()) return;

  if (state.IsChanceNode()) {
    for (const auto& [action, prob] : state.ChanceOutcomes()) {
      auto child = state.Clone();
      child->ApplyAction(action);
      CollectInfoStateStringsBeforeRoundRecursive(
          *child, current_round + 1, target_round, result);
    }
  } else {
    if (current_round < target_round) {
      Player player = state.CurrentPlayer();
      result[player].insert(state.InformationStateString(player));
    }
    for (Action action : state.LegalActions()) {
      auto child = state.Clone();
      child->ApplyAction(action);
      CollectInfoStateStringsBeforeRoundRecursive(
          *child, current_round, target_round, result);
    }
  }
}

}  // namespace

std::vector<std::unique_ptr<State>> CollectStates(
    const Game& game,
    const std::function<bool(const State&)>& predicate) {
  std::vector<std::unique_ptr<State>> result;
  auto initial = game.NewInitialState();
  CollectStatesRecursive(*initial, predicate, result);
  return result;
}

std::vector<std::unique_ptr<State>> CollectStatesAtDepth(
    const Game& game, int depth) {
  std::vector<std::unique_ptr<State>> result;
  auto initial = game.NewInitialState();
  CollectStatesAtDepthRecursive(*initial, 0, depth, result);
  return result;
}

std::vector<std::unique_ptr<State>> CollectStatesAtRound(
    const Game& game, int round) {
  std::vector<std::unique_ptr<State>> result;
  auto initial = game.NewInitialState();
  CollectStatesAtRoundRecursive(*initial, 0, round, result);
  return result;
}

std::unordered_map<int, std::unordered_set<std::string>>
CollectInfoStateStringsBeforeRound(const Game& game, int round) {
  std::unordered_map<int, std::unordered_set<std::string>> result;
  auto initial = game.NewInitialState();
  CollectInfoStateStringsBeforeRoundRecursive(*initial, 0, round, result);
  return result;
}

std::unordered_map<int, std::unordered_set<std::string>>
CollectInfoStateStringsBeforeDepth(const Game& game, int depth_limit) {
  std::unordered_map<int, std::unordered_set<std::string>> result;
  auto initial = game.NewInitialState();
  CollectInfoStateStringsBeforeDepthRecursive(*initial, 0, depth_limit, result);
  return result;
}

std::unordered_map<std::string, std::vector<std::unique_ptr<State>>>
GroupStatesByPublicObservation(
    const Game& game,
    std::vector<std::unique_ptr<State>> states) {
  auto observer = game.MakeObserver(kPublicStateObsType, {});
  std::unordered_map<std::string, std::vector<std::unique_ptr<State>>> grouped;
  for (auto& state : states) {
    std::string pub_obs = observer->StringFrom(*state, kDefaultPlayerId);
    grouped[pub_obs].push_back(std::move(state));
  }
  return grouped;
}

std::unordered_map<std::string, double> ComputeCounterfactualValuesAtStates(
    const Game& game,
    const Policy& policy,
    Player player,
    const std::vector<const State*>& states,
    const std::unordered_map<std::string, double>& reach_probs) {
  std::unordered_map<std::string, double> cf_values;

  for (const State* state : states) {
    std::string info_state = state->InformationStateString(player);
    std::string hist = state->HistoryString();

    double reach = 0.0;
    auto it = reach_probs.find(hist);
    if (it != reach_probs.end()) {
      reach = it->second;
    }

    auto returns = algorithms::ExpectedReturns(*state, policy, -1, false);
    double value = returns[player];

    cf_values[info_state] += reach * value;
  }

  return cf_values;
}

std::unordered_map<std::string, double> ComputeCounterfactualValuesAtStates(
    const Game& game,
    const Policy& policy,
    Player player,
    const std::vector<const State*>& states) {
  std::unordered_map<std::string, double> cf_values;

  for (const State* state : states) {
    std::string info_state = state->InformationStateString(player);
    if (cf_values.find(info_state) == cf_values.end()) {
      auto returns = algorithms::ExpectedReturns(*state, policy, -1, false);
      cf_values[info_state] = returns[player];
    }
  }

  return cf_values;
}

std::unordered_map<std::string, double> ComputeReachProbabilities(
    const Game& game,
    const Policy& opponent_policy,
    Player opponent,
    const std::vector<const State*>& states) {
  std::unordered_set<std::string> target_histories;
  for (const State* s : states) {
    target_histories.insert(s->HistoryString());
  }

  std::unordered_map<std::string, double> result;
  auto initial = game.NewInitialState();
  ComputeReachProbsRecursive(*initial, opponent_policy, opponent,
                              1.0, target_histories, result);

  return result;
}

std::array<std::unordered_set<std::string>, 2>
CollectSubgameInfoStatesPerPlayer(
    const std::vector<std::unique_ptr<State>>& roots) {
  std::array<std::unordered_set<std::string>, 2> result;
  std::function<void(const State&)> traverse = [&](const State& state) {
    if (state.IsTerminal()) return;
    if (state.IsChanceNode()) {
      for (const auto& [action, prob] : state.ChanceOutcomes()) {
        auto child = state.Clone();
        child->ApplyAction(action);
        traverse(*child);
      }
    } else {
      Player p = state.CurrentPlayer();
      result[p].insert(state.InformationStateString(p));
      for (Action action : state.LegalActions()) {
        auto child = state.Clone();
        child->ApplyAction(action);
        traverse(*child);
      }
    }
  };
  for (const auto& root : roots) traverse(*root);
  return result;
}

std::vector<SubgameRoot> BuildSubgameRoots(
    const std::vector<std::unique_ptr<State>>& states,
    Player resolving_player,
    const std::unordered_map<std::string, double>& reach_probs) {
  std::vector<SubgameRoot> roots;
  for (const auto& state : states) {
    std::string hist = state->HistoryString();
    auto it = reach_probs.find(hist);
    if (it == reach_probs.end() || it->second <= 0) continue;

    SubgameRoot root;
    root.state = state->Clone();
    root.info_state_string = state->InformationStateString(resolving_player);
    root.reach_prob = it->second;
    roots.push_back(std::move(root));
  }
  return roots;
}

SubgameDecomposition DecomposeGameAtRound(
    std::shared_ptr<const Game> game,
    const Policy& policy,
    int round) {
  SubgameDecomposition decomp;
  decomp.game = game;
  decomp.trunk_info_states = CollectInfoStateStringsBeforeRound(*game, round);

  auto roots = CollectStatesAtRound(*game, round);
  std::vector<const State*> ptrs;
  for (const auto& s : roots) ptrs.push_back(s.get());

  decomp.reach_probs[0] = ComputeReachProbabilities(*game, policy, 0, ptrs);
  decomp.reach_probs[1] = ComputeReachProbabilities(*game, policy, 1, ptrs);
  decomp.cfvs[0] = ComputeCounterfactualValuesAtStates(
      *game, policy, 0, ptrs, decomp.reach_probs[1]);
  decomp.cfvs[1] = ComputeCounterfactualValuesAtStates(
      *game, policy, 1, ptrs, decomp.reach_probs[0]);

  decomp.grouped_subgames =
      GroupStatesByPublicObservation(*game, std::move(roots));
  return decomp;
}

SubgameDecomposition DecomposeGameAtDepth(
    std::shared_ptr<const Game> game,
    const Policy& policy,
    int depth) {
  SubgameDecomposition decomp;
  decomp.game = game;
  decomp.trunk_info_states = CollectInfoStateStringsBeforeDepth(*game, depth);

  auto roots = CollectStatesAtDepth(*game, depth);
  std::vector<const State*> ptrs;
  for (const auto& s : roots) ptrs.push_back(s.get());

  decomp.reach_probs[0] = ComputeReachProbabilities(*game, policy, 0, ptrs);
  decomp.reach_probs[1] = ComputeReachProbabilities(*game, policy, 1, ptrs);
  decomp.cfvs[0] = ComputeCounterfactualValuesAtStates(
      *game, policy, 0, ptrs, decomp.reach_probs[1]);
  decomp.cfvs[1] = ComputeCounterfactualValuesAtStates(
      *game, policy, 1, ptrs, decomp.reach_probs[0]);

  decomp.grouped_subgames =
      GroupStatesByPublicObservation(*game, std::move(roots));
  return decomp;
}

SubgameDecomposition DecomposeGameStructureAtRound(
    std::shared_ptr<const Game> game,
    int round) {
  SubgameDecomposition decomp;
  decomp.game = game;
  decomp.trunk_info_states = CollectInfoStateStringsBeforeRound(*game, round);
  auto roots = CollectStatesAtRound(*game, round);
  decomp.grouped_subgames =
      GroupStatesByPublicObservation(*game, std::move(roots));
  // reach_probs and cfvs left empty — caller fills from their own solve
  return decomp;
}

SubgameDecomposition DecomposeGameStructureAtDepth(
    std::shared_ptr<const Game> game,
    int depth) {
  SubgameDecomposition decomp;
  decomp.game = game;
  decomp.trunk_info_states = CollectInfoStateStringsBeforeDepth(*game, depth);
  auto roots = CollectStatesAtDepth(*game, depth);
  decomp.grouped_subgames =
      GroupStatesByPublicObservation(*game, std::move(roots));
  // reach_probs and cfvs left empty — caller fills from their own solve
  return decomp;
}

}  // namespace open_spiel
