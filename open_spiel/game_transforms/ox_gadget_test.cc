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

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/algorithms/best_response.h"
#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/game_transforms/subgame_utils.h"
#include "open_spiel/game_transforms/turn_based_simultaneous_game.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace {

// Helper: build OX SubgameRoots (non-resolving player's IS, resolving reach)
std::vector<SubgameRoot> BuildOXRoots(
    const std::vector<std::unique_ptr<State>>& states,
    Player non_resolving,
    const std::unordered_map<std::string, double>& resolving_reach) {
  return BuildSubgameRoots(states, non_resolving, resolving_reach);
}

// Helper: compute model info set reach (non-resolving player's info set reach
// under the opponent model)
std::unordered_map<std::string, double> ComputeModelInfoSetReach(
    const Game& game,
    const Policy& opponent_model,
    Player non_resolving,
    const std::vector<std::unique_ptr<State>>& roots) {
  std::vector<const State*> ptrs;
  for (const auto& r : roots) ptrs.push_back(r.get());

  auto model_reach = ComputeReachProbabilities(
      game, opponent_model, non_resolving, ptrs);

  std::unordered_map<std::string, double> result;
  for (const auto& root : roots) {
    std::string is = root->InformationStateString(non_resolving);
    auto it = model_reach.find(root->HistoryString());
    if (it != model_reach.end()) {
      result[is] += it->second;
    }
  }
  return result;
}

// Helper: compute CBV for non-resolving player
std::unordered_map<std::string, double> ComputeCBV(
    const Game& game,
    const Policy& blueprint,
    Player non_resolving,
    const std::vector<std::unique_ptr<State>>& roots,
    const std::unordered_map<std::string, double>& resolving_reach) {
  algorithms::TabularBestResponse br(game, non_resolving, &blueprint);

  std::unordered_map<std::string, double> cbv_values;
  for (const auto& root : roots) {
    std::string info_state = root->InformationStateString(non_resolving);
    std::string hist = root->HistoryString();
    auto reach_it = resolving_reach.find(hist);
    double reach = (reach_it != resolving_reach.end()) ? reach_it->second : 0.0;
    if (reach > 0) {
      cbv_values[info_state] += reach * br.Value(hist);
    }
  }
  return cbv_values;
}

// =============================================================================
// TestOXConstruction
// =============================================================================

void TestOXConstruction() {
  std::cout << "TestOXConstruction..." << std::endl;

  auto game = LoadGame("kuhn_poker");
  auto uniform = std::make_shared<UniformPolicy>();

  auto states = CollectStatesAtDepth(*game, 2);
  std::vector<const State*> ptrs;
  for (const auto& s : states) ptrs.push_back(s.get());

  // For OX with resolving=P1 (res=1, non_res=0):
  // - info_state_string = P0's (non-resolving) info state
  // - reach_prob = P1's (resolving) reach
  auto resolving_reach = ComputeReachProbabilities(*game, *uniform, 1, ptrs);
  auto cbv_values = ComputeCBV(*game, *uniform, 0, states, resolving_reach);
  auto ox_roots = BuildOXRoots(states, 0, resolving_reach);
  SPIEL_CHECK_GT(ox_roots.size(), 0);

  auto model_reach = ComputeModelInfoSetReach(*game, *uniform, 0, states);

  double beta = 5.0;
  auto ox = CreateOXGadgetGame(game, std::move(ox_roots), 1,
                                cbv_values, beta, model_reach);

  SPIEL_CHECK_EQ(ox->NumPlayers(), 2);
  SPIEL_CHECK_EQ(ox->ResolvingPlayer(), 1);
  SPIEL_CHECK_EQ(ox->NonResolvingPlayer(), 0);
  SPIEL_CHECK_GT(ox->NumInfoSets(), 0);
  SPIEL_CHECK_FLOAT_NEAR(ox->Beta(), 5.0, 1e-9);

  std::cout << "  " << ox->NumSubgameRoots() << " roots, "
            << ox->NumInfoSets() << " info sets" << std::endl;
  std::cout << "TestOXConstruction PASSED" << std::endl;
}

// =============================================================================
// TestOXBranchProbabilities
// =============================================================================

void TestOXBranchProbabilities() {
  std::cout << "TestOXBranchProbabilities..." << std::endl;

  auto game = LoadGame("kuhn_poker");
  auto uniform = std::make_shared<UniformPolicy>();

  // We need exactly 3 info sets for P0 at depth 2 in Kuhn
  // (P0 sees cards J, Q, K after betting/checking)
  auto states = CollectStatesAtDepth(*game, 2);
  std::vector<const State*> ptrs;
  for (const auto& s : states) ptrs.push_back(s.get());

  auto resolving_reach = ComputeReachProbabilities(*game, *uniform, 1, ptrs);
  auto cbv_values = ComputeCBV(*game, *uniform, 0, states, resolving_reach);
  auto ox_roots = BuildOXRoots(states, 0, resolving_reach);
  auto model_reach = ComputeModelInfoSetReach(*game, *uniform, 0, states);

  double beta = 5.0;
  auto ox = CreateOXGadgetGame(game, std::move(ox_roots), 1,
                                cbv_values, beta, model_reach);

  int k = ox->NumInfoSets();
  std::cout << "  k = " << k << std::endl;

  double expected_exploit = 1.0 / (k * beta + 1.0);
  double expected_safety = (k * beta) / (k * beta + 1.0);

  SPIEL_CHECK_FLOAT_NEAR(ox->ExploitBranchProb(), expected_exploit, 1e-9);
  SPIEL_CHECK_FLOAT_NEAR(ox->SafetyBranchProb(), expected_safety, 1e-9);
  SPIEL_CHECK_FLOAT_NEAR(ox->ExploitBranchProb() + ox->SafetyBranchProb(),
                          1.0, 1e-9);

  std::cout << "  Exploit prob = " << ox->ExploitBranchProb()
            << " (expected " << expected_exploit << ")" << std::endl;
  std::cout << "  Safety prob = " << ox->SafetyBranchProb()
            << " (expected " << expected_safety << ")" << std::endl;
  std::cout << "TestOXBranchProbabilities PASSED" << std::endl;
}

// =============================================================================
// TestOXStateTransitions
// =============================================================================

void TestOXStateTransitions() {
  std::cout << "TestOXStateTransitions..." << std::endl;

  auto game = LoadGame("kuhn_poker");
  auto uniform = std::make_shared<UniformPolicy>();

  auto states = CollectStatesAtDepth(*game, 2);
  std::vector<const State*> ptrs;
  for (const auto& s : states) ptrs.push_back(s.get());

  auto resolving_reach = ComputeReachProbabilities(*game, *uniform, 1, ptrs);
  auto cbv_values = ComputeCBV(*game, *uniform, 0, states, resolving_reach);
  auto ox_roots = BuildOXRoots(states, 0, resolving_reach);
  auto model_reach = ComputeModelInfoSetReach(*game, *uniform, 0, states);

  double beta = 5.0;
  auto ox = CreateOXGadgetGame(game, std::move(ox_roots), 1,
                                cbv_values, beta, model_reach);

  // --- Safety Branch: InitialChance → kSafeChanceInfoSet → kOptionChoice
  //     → [enter] → kChanceWithinInfoSet → kSubgame ---
  {
    auto state = ox->NewInitialState();

    // Phase kInitialChance: chance node
    SPIEL_CHECK_EQ(state->CurrentPlayer(), kChancePlayerId);
    auto outcomes = state->ChanceOutcomes();
    SPIEL_CHECK_EQ(static_cast<int>(outcomes.size()), 2);
    double total_prob = 0.0;
    for (const auto& [a, p] : outcomes) total_prob += p;
    SPIEL_CHECK_FLOAT_NEAR(total_prob, 1.0, 1e-9);

    // Take safety branch (action 0)
    state->ApplyAction(0);

    // Phase kSafeChanceInfoSet: CHANCE picks IS uniformly (not player!)
    SPIEL_CHECK_EQ(state->CurrentPlayer(), kChancePlayerId);
    auto safe_outcomes = state->ChanceOutcomes();
    SPIEL_CHECK_EQ(static_cast<int>(safe_outcomes.size()), ox->NumInfoSets());
    double safe_prob_sum = 0.0;
    for (const auto& [a, p] : safe_outcomes) safe_prob_sum += p;
    SPIEL_CHECK_FLOAT_NEAR(safe_prob_sum, 1.0, 1e-9);
    // All probabilities should be equal (uniform)
    double expected_uniform = 1.0 / ox->NumInfoSets();
    for (const auto& [a, p] : safe_outcomes) {
      SPIEL_CHECK_FLOAT_NEAR(p, expected_uniform, 1e-9);
    }

    // Resolving player sees "ox_start"
    SPIEL_CHECK_EQ(state->InformationStateString(1), "ox_start");
    SPIEL_CHECK_EQ(state->InformationStateString(0), "ox_safe:chance");

    // Chance picks first info set
    state->ApplyAction(safe_outcomes[0].first);

    // Phase kOptionChoice: NON-resolving player (P0) makes enter/out choice
    SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);  // non-resolving player
    auto option_actions = state->LegalActions();
    SPIEL_CHECK_EQ(static_cast<int>(option_actions.size()), 2);

    // Resolving player sees "ox_start"
    SPIEL_CHECK_EQ(state->InformationStateString(1), "ox_start");
    // Non-resolving player sees option info state
    std::string non_res_is = state->InformationStateString(0);
    SPIEL_CHECK_TRUE(non_res_is.find("ox_option:") == 0);

    // Choose "enter" (action 1)
    state->ApplyAction(OXGadgetGame::kEnterAction);

    // Phase kChanceWithinInfoSet: chance picks state within IS
    SPIEL_CHECK_EQ(state->CurrentPlayer(), kChancePlayerId);
    auto within_outcomes = state->ChanceOutcomes();
    double prob_sum = 0.0;
    for (const auto& [a, p] : within_outcomes) prob_sum += p;
    SPIEL_CHECK_FLOAT_NEAR(prob_sum, 1.0, 1e-9);

    // Resolving player still sees "ox_start"
    SPIEL_CHECK_EQ(state->InformationStateString(1), "ox_start");

    // Enter subgame
    state->ApplyAction(within_outcomes[0].first);
    if (!state->IsTerminal()) {
      // Subgame info states should be "ox_F:subgame:..."
      Player cp = state->CurrentPlayer();
      if (cp >= 0) {
        std::string is = state->InformationStateString(cp);
        SPIEL_CHECK_TRUE(is.find("ox_F:subgame:") == 0);
      }
    }
  }

  // --- Safety Branch: "out" choice → Terminal with {0, 0} ---
  {
    auto state = ox->NewInitialState();
    state->ApplyAction(0);  // safety branch

    // Chance picks IS
    auto safe_outcomes = state->ChanceOutcomes();
    state->ApplyAction(safe_outcomes[0].first);

    // Non-resolving player chooses "out"
    SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
    state->ApplyAction(OXGadgetGame::kOutAction);

    // Should be terminal now
    SPIEL_CHECK_TRUE(state->IsTerminal());
    auto returns = state->Returns();
    SPIEL_CHECK_FLOAT_NEAR(returns[0], 0.0, 1e-9);
    SPIEL_CHECK_FLOAT_NEAR(returns[1], 0.0, 1e-9);
  }

  // --- Exploit Branch: InitialChance → kExploitChanceInfoSet
  //     → kChanceWithinInfoSet → kSubgame ---
  {
    auto state = ox->NewInitialState();

    // Take exploit branch (action 1)
    state->ApplyAction(1);

    // Phase kExploitChanceInfoSet: chance picks IS by model reach
    SPIEL_CHECK_EQ(state->CurrentPlayer(), kChancePlayerId);
    SPIEL_CHECK_EQ(state->InformationStateString(1), "ox_start");
    SPIEL_CHECK_EQ(state->InformationStateString(0), "ox_exploit:chance");

    auto info_set_outcomes = state->ChanceOutcomes();
    double total = 0.0;
    for (const auto& [a, p] : info_set_outcomes) total += p;
    SPIEL_CHECK_FLOAT_NEAR(total, 1.0, 1e-9);

    // Pick first info set
    state->ApplyAction(info_set_outcomes[0].first);

    // Phase kChanceWithinInfoSet (NO option choice in exploit branch!)
    SPIEL_CHECK_EQ(state->CurrentPlayer(), kChancePlayerId);
    SPIEL_CHECK_EQ(state->InformationStateString(1), "ox_start");

    auto within = state->ChanceOutcomes();
    state->ApplyAction(within[0].first);

    if (!state->IsTerminal()) {
      Player cp = state->CurrentPlayer();
      if (cp >= 0) {
        std::string is = state->InformationStateString(cp);
        SPIEL_CHECK_TRUE(is.find("ox_F:subgame:") == 0);
      }
    }
  }

  std::cout << "TestOXStateTransitions PASSED" << std::endl;
}

// =============================================================================
// TestOXPayoffs
// =============================================================================

void TestOXPayoffs() {
  std::cout << "TestOXPayoffs..." << std::endl;

  auto game = LoadGame("kuhn_poker");
  auto uniform = std::make_shared<UniformPolicy>();

  auto states = CollectStatesAtDepth(*game, 2);
  std::vector<const State*> ptrs;
  for (const auto& s : states) ptrs.push_back(s.get());

  auto resolving_reach = ComputeReachProbabilities(*game, *uniform, 1, ptrs);
  auto cbv_values = ComputeCBV(*game, *uniform, 0, states, resolving_reach);
  auto ox_roots = BuildOXRoots(states, 0, resolving_reach);
  auto model_reach = ComputeModelInfoSetReach(*game, *uniform, 0, states);

  auto ox = CreateOXGadgetGame(game, std::move(ox_roots), 1,
                                cbv_values, 5.0, model_reach);

  // Play through safety branch (enter) to terminal and verify zero-sum
  {
    auto state = ox->NewInitialState();
    state->ApplyAction(0);  // safety branch
    auto safe_outcomes = state->ChanceOutcomes();
    state->ApplyAction(safe_outcomes[0].first);  // chance picks IS

    // Non-resolving player enters
    SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
    state->ApplyAction(OXGadgetGame::kEnterAction);

    state->ApplyAction(state->ChanceOutcomes()[0].first);  // chance within IS
    while (!state->IsTerminal()) {
      if (state->IsChanceNode()) {
        state->ApplyAction(state->ChanceOutcomes()[0].first);
      } else {
        state->ApplyAction(state->LegalActions()[0]);
      }
    }

    auto returns = state->Returns();
    SPIEL_CHECK_FLOAT_NEAR(returns[0] + returns[1], 0.0, 1e-9);
    std::cout << "  Safety branch (enter): P0=" << returns[0]
              << ", P1=" << returns[1] << std::endl;
  }

  // Play through safety branch "out" - verify {0, 0}
  {
    auto state = ox->NewInitialState();
    state->ApplyAction(0);  // safety branch
    auto safe_outcomes = state->ChanceOutcomes();
    state->ApplyAction(safe_outcomes[0].first);  // chance picks IS

    // Non-resolving player chooses "out"
    SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
    state->ApplyAction(OXGadgetGame::kOutAction);

    SPIEL_CHECK_TRUE(state->IsTerminal());
    auto returns = state->Returns();
    SPIEL_CHECK_FLOAT_NEAR(returns[0], 0.0, 1e-9);
    SPIEL_CHECK_FLOAT_NEAR(returns[1], 0.0, 1e-9);
    std::cout << "  Safety branch (out): P0=" << returns[0]
              << ", P1=" << returns[1] << std::endl;
  }

  // Play through exploit branch to terminal and verify zero-sum
  {
    auto state = ox->NewInitialState();
    state->ApplyAction(1);  // exploit branch
    state->ApplyAction(state->ChanceOutcomes()[0].first);  // chance picks IS
    state->ApplyAction(state->ChanceOutcomes()[0].first);  // chance within IS
    while (!state->IsTerminal()) {
      if (state->IsChanceNode()) {
        state->ApplyAction(state->ChanceOutcomes()[0].first);
      } else {
        state->ApplyAction(state->LegalActions()[0]);
      }
    }

    auto returns = state->Returns();
    SPIEL_CHECK_FLOAT_NEAR(returns[0] + returns[1], 0.0, 1e-9);
    std::cout << "  Exploit branch: P0=" << returns[0]
              << ", P1=" << returns[1] << std::endl;
  }

  std::cout << "TestOXPayoffs PASSED" << std::endl;
}

// =============================================================================
// TestCFROnOX
// =============================================================================

void TestCFROnOX() {
  std::cout << "TestCFROnOX..." << std::endl;

  auto game = LoadGame("kuhn_poker");
  auto uniform = std::make_shared<UniformPolicy>();

  auto states = CollectStatesAtDepth(*game, 2);
  std::vector<const State*> ptrs;
  for (const auto& s : states) ptrs.push_back(s.get());

  auto resolving_reach = ComputeReachProbabilities(*game, *uniform, 1, ptrs);
  auto cbv_values = ComputeCBV(*game, *uniform, 0, states, resolving_reach);
  auto ox_roots = BuildOXRoots(states, 0, resolving_reach);
  auto model_reach = ComputeModelInfoSetReach(*game, *uniform, 0, states);

  auto ox = CreateOXGadgetGame(game, std::move(ox_roots), 1,
                                cbv_values, 5.0, model_reach);

  algorithms::CFRSolverBase cfr(*ox, true, true, true);
  for (int i = 0; i < 1000; ++i) cfr.EvaluateAndUpdatePolicy();

  double exp = algorithms::Exploitability(*ox, *cfr.AveragePolicy());
  std::cout << "  Exploitability = " << exp << std::endl;
  SPIEL_CHECK_LT(exp, 0.1);

  std::cout << "TestCFROnOX PASSED" << std::endl;
}

// =============================================================================
// TestOXLeducSafety
// =============================================================================

void TestOXLeducSafety() {
  std::cout << "TestOXLeducSafety..." << std::endl;

  auto game = LoadGame("leduc_poker");

  // Solve full game with CFR
  algorithms::CFRSolverBase solver(*game, true, true, true);
  for (int i = 0; i < 1000; ++i) solver.EvaluateAndUpdatePolicy();
  auto policy = solver.AveragePolicy();
  TabularPolicy trunk = solver.TabularAveragePolicy();
  double full_exp = algorithms::Exploitability(*game, *policy);

  // Decompose at round 3
  auto decomp = DecomposeGameAtRound(game, *policy, 3);

  // Uniform model as opponent model
  UniformPolicy uniform_model;

  // Test a few beta values: higher beta = safer
  double beta_values[] = {1.0, 5.0, 20.0};
  for (double beta : beta_values) {
    auto ox_policy =
        ResolveWithOXGadget(decomp, trunk, beta, uniform_model, 500);
    double ox_exp = algorithms::Exploitability(*game, *ox_policy);
    std::cout << "  beta=" << beta << " exploitability=" << ox_exp
              << std::endl;
    // OX should remain bounded (safe)
    SPIEL_CHECK_LT(ox_exp, 1.0);
  }

  std::cout << "  Full CFR exploitability: " << full_exp << std::endl;
  std::cout << "TestOXLeducSafety PASSED" << std::endl;
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::TestOXConstruction();
  open_spiel::TestOXBranchProbabilities();
  open_spiel::TestOXStateTransitions();
  open_spiel::TestOXPayoffs();
  open_spiel::TestCFROnOX();
  open_spiel::TestOXLeducSafety();

  std::cout << "\nAll OX tests passed!" << std::endl;
  return 0;
}
