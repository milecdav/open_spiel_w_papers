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

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "open_spiel/algorithms/best_response.h"
#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/algorithms/expected_returns.h"
#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/game_transforms/matrix_valued_states.h"
#include "open_spiel/game_transforms/resolving_gadget.h"
#include "open_spiel/game_transforms/subgame_utils.h"
#include "open_spiel/game_transforms/turn_based_simultaneous_game.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace {

// =============================================================================
// Helper: tabularize an opponent model for a game
// =============================================================================

TabularPolicy MakeUniformTabular(const Game& game) {
  TabularPolicy result;
  UniformPolicy uniform;
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
      auto ap = uniform.GetStatePolicy(state, pl);
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

// Compute the return of target_player when playing `target_policy` against
// `opponent_policy`. Uses per-player ExpectedReturns.
double GainAgainst(const Game& game, int target_player,
                   const Policy& target_policy,
                   const Policy& opponent_policy) {
  std::vector<const Policy*> policies(game.NumPlayers());
  policies[target_player] = &target_policy;
  policies[1 - target_player] = &opponent_policy;
  auto returns = algorithms::ExpectedReturns(
      *game.NewInitialState(), policies, -1);
  return returns[target_player];
}

// =============================================================================
// Test 1: GadgetPolicyWrapper
// =============================================================================

void TestGadgetPolicyWrapper() {
  std::cout << "TestGadgetPolicyWrapper..." << std::endl;

  // Create a simple tabular policy with known entries
  TabularPolicy underlying;
  underlying.SetStatePolicy("info_A", {{0, 0.3}, {1, 0.7}});
  underlying.SetStatePolicy("info_B", {{0, 0.5}, {1, 0.5}});

  // Test resolving gadget wrapper
  {
    GadgetPolicyWrapper wrapper(&underlying, "gadget_F:subgame:",
                                GadgetGame::kFollowAction, 0);

    // Subgame info states: should strip prefix and delegate
    auto ap = wrapper.GetStatePolicy("gadget_F:subgame:info_A");
    SPIEL_CHECK_EQ(ap.size(), 2);
    SPIEL_CHECK_FLOAT_NEAR(ap[0].second, 0.3, 1e-9);
    SPIEL_CHECK_FLOAT_NEAR(ap[1].second, 0.7, 1e-9);

    // T/F choice: should return Follow with prob 1.0
    auto tf = wrapper.GetStatePolicy("gadget_choice:some_info");
    SPIEL_CHECK_EQ(tf.size(), 2);
    SPIEL_CHECK_FLOAT_NEAR(tf[0].second, 0.0, 1e-9);  // T = 0
    SPIEL_CHECK_FLOAT_NEAR(tf[1].second, 1.0, 1e-9);  // F = 1

    // opponent_choosing: not a resolving player info state
    auto opp = wrapper.GetStatePolicy("gadget_choice:opponent_choosing");
    SPIEL_CHECK_TRUE(opp.empty());

    // Unknown info state: empty
    auto unk = wrapper.GetStatePolicy("unknown");
    SPIEL_CHECK_TRUE(unk.empty());
  }

  // Test max-margin gadget wrapper
  {
    GadgetPolicyWrapper wrapper(&underlying, "mm_F:subgame:", -1, 5);

    // Subgame info states
    auto ap = wrapper.GetStatePolicy("mm_F:subgame:info_B");
    SPIEL_CHECK_EQ(ap.size(), 2);
    SPIEL_CHECK_FLOAT_NEAR(ap[0].second, 0.5, 1e-9);

    // mm_start: uniform over 5 actions
    auto mm = wrapper.GetStatePolicy("mm_start");
    SPIEL_CHECK_EQ(mm.size(), 5);
    for (const auto& [a, p] : mm) {
      SPIEL_CHECK_FLOAT_NEAR(p, 0.2, 1e-9);
    }
  }

  // Test unsafe subgame wrapper
  {
    GadgetPolicyWrapper wrapper(&underlying, "unsafe:subgame:", -1, 0);

    auto ap = wrapper.GetStatePolicy("unsafe:subgame:info_A");
    SPIEL_CHECK_EQ(ap.size(), 2);
    SPIEL_CHECK_FLOAT_NEAR(ap[0].second, 0.3, 1e-9);
  }

  std::cout << "TestGadgetPolicyWrapper PASSED" << std::endl;
}

// =============================================================================
// Test 2: CDBR on Kuhn poker (best response against uniform)
// =============================================================================

void TestCDBRKuhn() {
  std::cout << "TestCDBRKuhn..." << std::endl;

  auto game = LoadGame("kuhn_poker");
  UniformPolicy uniform;
  TabularPolicy uniform_tabular = MakeUniformTabular(*game);

  // Verify tabular policy works
  {
    auto ret = algorithms::ExpectedReturns(
        *game->NewInitialState(), uniform_tabular, -1);
    std::cout << "  Uniform returns: " << ret[0] << ", " << ret[1] << std::endl;
  }

  // CDBR: RNR with p=1.0, no gadget
  ResolvingConfig config;
  config.solver = SolverType::kRNR;
  config.gadget = GadgetType::kNone;
  config.opponent_model = &uniform_tabular;
  config.p = 1.0;
  config.target_player = 0;
  config.cfr_iterations = 200;

  auto result = ContinualResolve(
      game, uniform, config, 2, MVSGame::DepthMode::kActionBased);

  // Debug: print policy table
  std::cout << "  Policy table size: " << result->PolicyTable().size()
            << std::endl;

  // Verify all entries have valid action probs
  std::function<void(const State&)> verify = [&](const State& state) {
    if (state.IsTerminal()) return;
    if (state.IsChanceNode()) {
      for (const auto& [a, p] : state.ChanceOutcomes()) {
        auto child = state.Clone();
        child->ApplyAction(a);
        verify(*child);
      }
      return;
    }
    Player pl = state.CurrentPlayer();
    std::string is = state.InformationStateString(pl);
    auto ap = result->GetStatePolicy(is);
    if (ap.empty()) {
      std::cout << "  MISSING: " << is << " (player " << pl << ")" << std::endl;
    } else {
      for (Action a : state.LegalActions()) {
        double prob = GetProb(ap, a);
        if (prob < 0) {
          std::cout << "  BAD ACTION " << a << " at " << is
                    << " (player " << pl << "), policy has: ";
          for (const auto& [pa, pp] : ap) {
            std::cout << pa << "=" << pp << " ";
          }
          std::cout << std::endl;
        }
      }
    }
    for (Action a : state.LegalActions()) {
      auto child = state.Clone();
      child->ApplyAction(a);
      verify(*child);
    }
  };
  verify(*game->NewInitialState());

  double exp = algorithms::Exploitability(*game, *result);
  std::cout << "  Exploitability: " << exp << std::endl;

  // Compute gain against uniform opponent (target uses result, opp uses uniform)
  double gain = GainAgainst(*game, 0, *result, uniform_tabular);

  std::cout << "  P0 gain vs uniform: " << gain << std::endl;
  // CDBR should exploit the uniform opponent significantly
  SPIEL_CHECK_GT(gain, 0.3);

  std::cout << "TestCDBRKuhn PASSED" << std::endl;
}

// =============================================================================
// Test 3: CDRNR on Kuhn poker with resolving gadget
// =============================================================================

void TestCDRNRKuhnResolving() {
  std::cout << "TestCDRNRKuhnResolving..." << std::endl;

  auto game = LoadGame("kuhn_poker");
  UniformPolicy uniform;
  TabularPolicy uniform_tabular = MakeUniformTabular(*game);

  ResolvingConfig config;
  config.solver = SolverType::kRNR;
  config.gadget = GadgetType::kResolving;
  config.opponent_model = &uniform_tabular;
  config.p = 0.5;
  config.target_player = 0;
  config.cfr_iterations = 300;

  auto result = ContinualResolve(
      game, uniform, config, 2, MVSGame::DepthMode::kActionBased);

  double gain = GainAgainst(*game, 0, *result, uniform_tabular);
  double exp = algorithms::Exploitability(*game, *result);

  std::cout << "  P0 gain vs uniform: " << gain << std::endl;
  std::cout << "  Exploitability: " << exp << std::endl;

  // With p=0.5, should have moderate exploitability and positive gain
  SPIEL_CHECK_LT(exp, 1.0);
  SPIEL_CHECK_GT(gain, -0.1);

  std::cout << "TestCDRNRKuhnResolving PASSED" << std::endl;
}

// =============================================================================
// Test 4: CDRNR on Kuhn poker with max-margin gadget
// =============================================================================

void TestCDRNRKuhnMaxMargin() {
  std::cout << "TestCDRNRKuhnMaxMargin..." << std::endl;

  auto game = LoadGame("kuhn_poker");
  UniformPolicy uniform;
  TabularPolicy uniform_tabular = MakeUniformTabular(*game);

  ResolvingConfig config;
  config.solver = SolverType::kRNR;
  config.gadget = GadgetType::kMaxMargin;
  config.opponent_model = &uniform_tabular;
  config.p = 0.5;
  config.target_player = 0;
  config.cfr_iterations = 300;

  auto result = ContinualResolve(
      game, uniform, config, 2, MVSGame::DepthMode::kActionBased);

  double gain = GainAgainst(*game, 0, *result, uniform_tabular);
  double exp = algorithms::Exploitability(*game, *result);

  std::cout << "  P0 gain vs uniform: " << gain << std::endl;
  std::cout << "  Exploitability: " << exp << std::endl;

  SPIEL_CHECK_LT(exp, 1.0);
  SPIEL_CHECK_GT(gain, -0.1);

  std::cout << "TestCDRNRKuhnMaxMargin PASSED" << std::endl;
}

// =============================================================================
// Test 5: CDRNR on Leduc poker with resolving gadget
// =============================================================================

void TestCDRNRLeducResolving() {
  std::cout << "TestCDRNRLeducResolving..." << std::endl;

  auto game = LoadGame("leduc_poker");
  UniformPolicy uniform;
  TabularPolicy uniform_tabular = MakeUniformTabular(*game);

  ResolvingConfig config;
  config.solver = SolverType::kRNR;
  config.gadget = GadgetType::kResolving;
  config.opponent_model = &uniform_tabular;
  config.p = 0.5;
  config.target_player = 0;
  config.cfr_iterations = 300;

  auto result = ContinualResolve(
      game, uniform, config, 3, MVSGame::DepthMode::kRoundBased);

  double gain = GainAgainst(*game, 0, *result, uniform_tabular);
  double exp = algorithms::Exploitability(*game, *result);

  std::cout << "  P0 gain vs uniform: " << gain << std::endl;
  std::cout << "  Exploitability: " << exp << std::endl;

  SPIEL_CHECK_GT(result->PolicyTable().size(), 0);
  SPIEL_CHECK_LT(exp, 2.0);
  SPIEL_CHECK_GT(gain, -0.5);

  std::cout << "TestCDRNRLeducResolving PASSED" << std::endl;
}

// =============================================================================
// Test 6: CDRNR on Leduc poker with max-margin gadget
// =============================================================================

void TestCDRNRLeducMaxMargin() {
  std::cout << "TestCDRNRLeducMaxMargin..." << std::endl;

  auto game = LoadGame("leduc_poker");
  UniformPolicy uniform;
  TabularPolicy uniform_tabular = MakeUniformTabular(*game);

  ResolvingConfig config;
  config.solver = SolverType::kRNR;
  config.gadget = GadgetType::kMaxMargin;
  config.opponent_model = &uniform_tabular;
  config.p = 0.5;
  config.target_player = 0;
  config.cfr_iterations = 300;

  auto result = ContinualResolve(
      game, uniform, config, 3, MVSGame::DepthMode::kRoundBased);

  double gain = GainAgainst(*game, 0, *result, uniform_tabular);
  double exp = algorithms::Exploitability(*game, *result);

  std::cout << "  P0 gain vs uniform: " << gain << std::endl;
  std::cout << "  Exploitability: " << exp << std::endl;

  SPIEL_CHECK_GT(result->PolicyTable().size(), 0);
  SPIEL_CHECK_LT(exp, 2.0);
  SPIEL_CHECK_GT(gain, -0.5);

  std::cout << "TestCDRNRLeducMaxMargin PASSED" << std::endl;
}

// =============================================================================
// Test 7: Sweep p on Kuhn - exploitability vs gain tradeoff
// =============================================================================

void TestSweepP() {
  std::cout << "TestSweepP..." << std::endl;

  auto game = LoadGame("kuhn_poker");
  UniformPolicy uniform;
  TabularPolicy uniform_tabular = MakeUniformTabular(*game);

  std::vector<double> ps = {0.0, 0.25, 0.5, 0.75, 1.0};

  for (int target = 0; target < 2; ++target) {
    std::cout << "  === Target player " << target << " ===" << std::endl;
    std::vector<double> exploitabilities;
    std::vector<double> gains_vs_uniform;

    for (double p : ps) {
      ResolvingConfig config;
      config.solver = SolverType::kRNR;
      config.gadget = GadgetType::kResolving;
      config.opponent_model = &uniform_tabular;
      config.p = p;
      config.target_player = target;
      config.cfr_iterations = 200;

      auto result = ContinualResolve(
          game, uniform, config, 2, MVSGame::DepthMode::kActionBased);

      double exp = algorithms::Exploitability(*game, *result);
      double gain = GainAgainst(*game, target, *result, uniform_tabular);

      exploitabilities.push_back(exp);
      gains_vs_uniform.push_back(gain);

      std::cout << "  p=" << p << ": gain_vs_uniform=" << gain
                << ", exploitability=" << exp << std::endl;
    }

    // At p=0 (Nash solving), exploitability should be low
    SPIEL_CHECK_LT(exploitabilities[0], 0.1);

    // At p=1, gain vs uniform should be high (exploiting the opponent)
    SPIEL_CHECK_GT(gains_vs_uniform[4], 0.3);

    // Exploitability should generally increase with p
    SPIEL_CHECK_LT(exploitabilities[0], exploitabilities[4]);

    std::cout << "  Trend: gain " << gains_vs_uniform[0] << " -> "
              << gains_vs_uniform[4]
              << ", expl " << exploitabilities[0] << " -> "
              << exploitabilities[4] << std::endl;
  }

  std::cout << "TestSweepP PASSED" << std::endl;
}

// =============================================================================
// Test 8: Standard resolving via ResolveSubgames (kCFR + kResolving)
// =============================================================================

void TestResolveSubgamesCFR() {
  std::cout << "TestResolveSubgamesCFR..." << std::endl;

  auto game = LoadGame("kuhn_poker");

  // Solve full game for baseline and decomposition
  algorithms::CFRSolverBase solver(*game, true, true, true);
  for (int i = 0; i < 500; ++i) solver.EvaluateAndUpdatePolicy();
  auto policy = solver.AveragePolicy();
  TabularPolicy trunk = solver.TabularAveragePolicy();
  double full_exp = algorithms::Exploitability(*game, *policy);

  // Decompose at depth 2
  auto decomp = DecomposeGameAtDepth(game, *policy, 2);

  // Resolve with CFR + resolving gadget (same as ResolveWithGadget)
  ResolvingConfig config;
  config.solver = SolverType::kCFR;
  config.gadget = GadgetType::kResolving;
  config.cfr_iterations = 500;

  auto result = ResolveSubgames(decomp, trunk, config);
  double resolved_exp = algorithms::Exploitability(*game, *result);

  std::cout << "  Full:     " << full_exp << std::endl;
  std::cout << "  Resolved: " << resolved_exp << std::endl;

  // Resolved exploitability should be reasonable
  SPIEL_CHECK_LT(resolved_exp, 0.5);

  std::cout << "TestResolveSubgamesCFR PASSED" << std::endl;
}

// =============================================================================
// Test 9: CDRNR p=0 vs Gadget resolve (should be equivalent)
// =============================================================================

void TestCDRNR_p0_vs_Gadget() {
  std::cout << "TestCDRNR_p0_vs_Gadget..." << std::endl;

  auto game = LoadGame("kuhn_poker");
  const int iters = 500;

  // --- Baseline: standard gadget resolve (CFR on full game + gadget) ---
  algorithms::CFRSolverBase cfr_solver(*game, true, true, true);
  for (int i = 0; i < iters; ++i) cfr_solver.EvaluateAndUpdatePolicy();
  auto cfr_policy = cfr_solver.AveragePolicy();
  TabularPolicy cfr_trunk = cfr_solver.TabularAveragePolicy();

  auto decomp = DecomposeGameAtDepth(game, *cfr_policy, 2);

  ResolvingConfig gadget_config;
  gadget_config.solver = SolverType::kCFR;
  gadget_config.gadget = GadgetType::kResolving;
  gadget_config.cfr_iterations = iters;

  auto gadget_result = ResolveSubgames(decomp, cfr_trunk, gadget_config);
  double gadget_exp = algorithms::Exploitability(*game, *gadget_result);

  UniformPolicy uniform;
  TabularPolicy uniform_tabular = MakeUniformTabular(*game);

  // --- Test CDRNR p=0 for both players ---
  for (int target = 0; target < 2; ++target) {
    double gadget_gain = GainAgainst(
        *game, target, *gadget_result, uniform_tabular);

    ResolvingConfig cdrnr_config;
    cdrnr_config.solver = SolverType::kRNR;
    cdrnr_config.gadget = GadgetType::kResolving;
    cdrnr_config.opponent_model = &uniform_tabular;
    cdrnr_config.p = 0.0;
    cdrnr_config.target_player = target;
    cdrnr_config.cfr_iterations = iters;

    auto cdrnr_result = ContinualResolve(
        game, uniform, cdrnr_config, 2, MVSGame::DepthMode::kActionBased);
    double cdrnr_exp = algorithms::Exploitability(*game, *cdrnr_result);
    double cdrnr_gain = GainAgainst(
        *game, target, *cdrnr_result, uniform_tabular);

    std::cout << "  [P" << target << "] Gadget: expl=" << gadget_exp
              << ", gain=" << gadget_gain << std::endl;
    std::cout << "  [P" << target << "] CDRNR:  expl=" << cdrnr_exp
              << ", gain=" << cdrnr_gain << std::endl;

    // With p=0, CDRNR should produce Nash-like exploitability
    SPIEL_CHECK_LT(cdrnr_exp, 0.05);
    SPIEL_CHECK_LT(std::abs(cdrnr_exp - gadget_exp), 0.05);
  }

  std::cout << "TestCDRNR_p0_vs_Gadget PASSED" << std::endl;
}

// =============================================================================
// Test 10: CDRNR p=1 vs best response (should be equivalent)
// =============================================================================

void TestCDRNR_p1_vs_BR() {
  std::cout << "TestCDRNR_p1_vs_BR..." << std::endl;

  auto game = LoadGame("kuhn_poker");
  UniformPolicy uniform;
  TabularPolicy uniform_tabular = MakeUniformTabular(*game);
  const int iters = 500;

  for (int target = 0; target < 2; ++target) {
    // --- Baseline: tabular best response against uniform ---
    algorithms::TabularBestResponse br(*game, target, &uniform_tabular);
    TabularPolicy br_policy = br.GetBestResponsePolicy();
    double br_gain = GainAgainst(*game, target, br_policy, uniform_tabular);
    double br_exp = algorithms::Exploitability(*game, br_policy);

    // --- CDRNR with p=1 (should be close to BR) ---
    ResolvingConfig cdrnr_config;
    cdrnr_config.solver = SolverType::kRNR;
    cdrnr_config.gadget = GadgetType::kNone;  // No safety, pure exploitation
    cdrnr_config.opponent_model = &uniform_tabular;
    cdrnr_config.p = 1.0;
    cdrnr_config.target_player = target;
    cdrnr_config.cfr_iterations = iters;

    auto cdrnr_result = ContinualResolve(
        game, uniform, cdrnr_config, 2, MVSGame::DepthMode::kActionBased);
    double cdrnr_gain = GainAgainst(
        *game, target, *cdrnr_result, uniform_tabular);
    double cdrnr_exp = algorithms::Exploitability(*game, *cdrnr_result);

    std::cout << "  [P" << target << "] BR:    gain=" << br_gain
              << ", expl=" << br_exp << std::endl;
    std::cout << "  [P" << target << "] CDRNR: gain=" << cdrnr_gain
              << ", expl=" << cdrnr_exp << std::endl;
    std::cout << "  [P" << target << "] Gain diff: "
              << std::abs(cdrnr_gain - br_gain)
              << ", Expl diff: " << std::abs(cdrnr_exp - br_exp) << std::endl;

    // CDRNR with p=1 should exploit uniform similarly to full BR
    SPIEL_CHECK_GT(cdrnr_gain, br_gain - 0.1);
  }

  std::cout << "TestCDRNR_p1_vs_BR PASSED" << std::endl;
}

// =============================================================================
// Test 11: CDRNR on Goofspiel(4) — both players, p sweep
// =============================================================================

void TestCDRNRGoofspiel() {
  std::cout << "TestCDRNRGoofspiel..." << std::endl;

  auto game = LoadGameAsTurnBased(
      "goofspiel",
      {{"num_cards", GameParameter(4)},
       {"imp_info", GameParameter(true)},
       {"points_order", GameParameter(std::string("descending"))}});
  UniformPolicy uniform;
  TabularPolicy uniform_tabular = MakeUniformTabular(*game);

  const int iters = 300;
  const int depth = 4;
  auto depth_mode = MVSGame::DepthMode::kActionBased;

  for (int target = 0; target < 2; ++target) {
    std::cout << "  === Target player " << target << " ===" << std::endl;

    // p=0: should give low exploitability (Nash-like)
    {
      ResolvingConfig config;
      config.solver = SolverType::kRNR;
      config.gadget = GadgetType::kResolving;
      config.opponent_model = &uniform_tabular;
      config.p = 0.0;
      config.target_player = target;
      config.cfr_iterations = iters;

      auto result = ContinualResolve(game, uniform, config, depth, depth_mode);
      double exp = algorithms::Exploitability(*game, *result);
      double gain = GainAgainst(*game, target, *result, uniform_tabular);

      std::cout << "  p=0: expl=" << exp << ", gain=" << gain << std::endl;
      SPIEL_CHECK_LT(exp, 0.15);
    }

    // p=0.5: moderate tradeoff
    {
      ResolvingConfig config;
      config.solver = SolverType::kRNR;
      config.gadget = GadgetType::kResolving;
      config.opponent_model = &uniform_tabular;
      config.p = 0.5;
      config.target_player = target;
      config.cfr_iterations = iters;

      auto result = ContinualResolve(game, uniform, config, depth, depth_mode);
      double exp = algorithms::Exploitability(*game, *result);
      double gain = GainAgainst(*game, target, *result, uniform_tabular);

      std::cout << "  p=0.5: expl=" << exp << ", gain=" << gain << std::endl;
      SPIEL_CHECK_GT(gain, -1.0);
    }

    // p=1: max exploitation
    {
      ResolvingConfig config;
      config.solver = SolverType::kRNR;
      config.gadget = GadgetType::kNone;
      config.opponent_model = &uniform_tabular;
      config.p = 1.0;
      config.target_player = target;
      config.cfr_iterations = iters;

      auto result = ContinualResolve(game, uniform, config, depth, depth_mode);
      double exp = algorithms::Exploitability(*game, *result);
      double gain = GainAgainst(*game, target, *result, uniform_tabular);

      std::cout << "  p=1: expl=" << exp << ", gain=" << gain << std::endl;
      SPIEL_CHECK_GT(gain, 0.0);
    }
  }

  std::cout << "TestCDRNRGoofspiel PASSED" << std::endl;
}

// =============================================================================
// Test 12: CDRNR on Leduc poker — both players, p sweep
// =============================================================================

void TestCDRNRLeduc() {
  std::cout << "TestCDRNRLeduc..." << std::endl;

  auto game = LoadGame("leduc_poker");
  UniformPolicy uniform;
  TabularPolicy uniform_tabular = MakeUniformTabular(*game);

  const int iters = 300;
  const int depth = 3;
  auto depth_mode = MVSGame::DepthMode::kRoundBased;

  for (int target = 0; target < 2; ++target) {
    std::cout << "  === Target player " << target << " ===" << std::endl;

    // p=0: should give low exploitability (Nash-like)
    {
      ResolvingConfig config;
      config.solver = SolverType::kRNR;
      config.gadget = GadgetType::kResolving;
      config.opponent_model = &uniform_tabular;
      config.p = 0.0;
      config.target_player = target;
      config.cfr_iterations = iters;

      auto result = ContinualResolve(game, uniform, config, depth, depth_mode);
      double exp = algorithms::Exploitability(*game, *result);
      double gain = GainAgainst(*game, target, *result, uniform_tabular);

      std::cout << "  p=0: expl=" << exp << ", gain=" << gain << std::endl;
      SPIEL_CHECK_LT(exp, 0.5);
    }

    // p=0.5: moderate tradeoff
    {
      ResolvingConfig config;
      config.solver = SolverType::kRNR;
      config.gadget = GadgetType::kResolving;
      config.opponent_model = &uniform_tabular;
      config.p = 0.5;
      config.target_player = target;
      config.cfr_iterations = iters;

      auto result = ContinualResolve(game, uniform, config, depth, depth_mode);
      double exp = algorithms::Exploitability(*game, *result);
      double gain = GainAgainst(*game, target, *result, uniform_tabular);

      std::cout << "  p=0.5: expl=" << exp << ", gain=" << gain << std::endl;
    }

    // p=1: max exploitation
    {
      ResolvingConfig config;
      config.solver = SolverType::kRNR;
      config.gadget = GadgetType::kNone;
      config.opponent_model = &uniform_tabular;
      config.p = 1.0;
      config.target_player = target;
      config.cfr_iterations = iters;

      auto result = ContinualResolve(game, uniform, config, depth, depth_mode);
      double exp = algorithms::Exploitability(*game, *result);
      double gain = GainAgainst(*game, target, *result, uniform_tabular);

      std::cout << "  p=1: expl=" << exp << ", gain=" << gain << std::endl;
      SPIEL_CHECK_GT(gain, 0.0);
    }
  }

  std::cout << "TestCDRNRLeduc PASSED" << std::endl;
}

// =============================================================================
// Test 13: Goofspiel p=0 vs Gadget resolve (should be equivalent)
// =============================================================================

void TestGoofspiel_p0_vs_Gadget() {
  std::cout << "TestGoofspiel_p0_vs_Gadget..." << std::endl;

  auto game = LoadGameAsTurnBased(
      "goofspiel",
      {{"num_cards", GameParameter(4)},
       {"imp_info", GameParameter(true)},
       {"points_order", GameParameter(std::string("descending"))}});
  const int iters = 300;
  const int depth = 4;

  // --- Baseline: standard gadget resolve (CFR on full game + gadget) ---
  algorithms::CFRSolverBase cfr_solver(*game, true, true, true);
  for (int i = 0; i < iters; ++i) cfr_solver.EvaluateAndUpdatePolicy();
  auto cfr_policy = cfr_solver.AveragePolicy();
  TabularPolicy cfr_trunk = cfr_solver.TabularAveragePolicy();

  auto decomp = DecomposeGameAtDepth(game, *cfr_policy, depth);

  ResolvingConfig gadget_config;
  gadget_config.solver = SolverType::kCFR;
  gadget_config.gadget = GadgetType::kResolving;
  gadget_config.cfr_iterations = iters;

  auto gadget_result = ResolveSubgames(decomp, cfr_trunk, gadget_config);
  double gadget_exp = algorithms::Exploitability(*game, *gadget_result);

  UniformPolicy uniform;
  TabularPolicy uniform_tabular = MakeUniformTabular(*game);

  // --- Test CDRNR p=0 for both players ---
  for (int target = 0; target < 2; ++target) {
    double gadget_gain = GainAgainst(
        *game, target, *gadget_result, uniform_tabular);

    ResolvingConfig cdrnr_config;
    cdrnr_config.solver = SolverType::kRNR;
    cdrnr_config.gadget = GadgetType::kResolving;
    cdrnr_config.opponent_model = &uniform_tabular;
    cdrnr_config.p = 0.0;
    cdrnr_config.target_player = target;
    cdrnr_config.cfr_iterations = iters;

    auto cdrnr_result = ContinualResolve(
        game, uniform, cdrnr_config, depth, MVSGame::DepthMode::kActionBased);
    double cdrnr_exp = algorithms::Exploitability(*game, *cdrnr_result);
    double cdrnr_gain = GainAgainst(
        *game, target, *cdrnr_result, uniform_tabular);

    std::cout << "  [P" << target << "] Gadget: expl=" << gadget_exp
              << ", gain=" << gadget_gain << std::endl;
    std::cout << "  [P" << target << "] CDRNR:  expl=" << cdrnr_exp
              << ", gain=" << cdrnr_gain << std::endl;

    // CDRNR p=0 should produce similar exploitability as gadget resolve
    SPIEL_CHECK_LT(cdrnr_exp, 0.15);
    SPIEL_CHECK_LT(std::abs(cdrnr_exp - gadget_exp), 0.1);
  }

  std::cout << "TestGoofspiel_p0_vs_Gadget PASSED" << std::endl;
}

// =============================================================================
// Test 14: Goofspiel p=1 vs best response (should be equivalent)
// =============================================================================

void TestGoofspiel_p1_vs_BR() {
  std::cout << "TestGoofspiel_p1_vs_BR..." << std::endl;

  auto game = LoadGameAsTurnBased(
      "goofspiel",
      {{"num_cards", GameParameter(4)},
       {"imp_info", GameParameter(true)},
       {"points_order", GameParameter(std::string("descending"))}});
  UniformPolicy uniform;
  TabularPolicy uniform_tabular = MakeUniformTabular(*game);
  const int iters = 300;
  const int depth = 4;

  for (int target = 0; target < 2; ++target) {
    // --- Baseline: tabular best response against uniform ---
    algorithms::TabularBestResponse br(*game, target, &uniform_tabular);
    TabularPolicy br_policy = br.GetBestResponsePolicy();
    double br_gain = GainAgainst(*game, target, br_policy, uniform_tabular);

    // --- CDRNR with p=1 (should be close to BR) ---
    ResolvingConfig cdrnr_config;
    cdrnr_config.solver = SolverType::kRNR;
    cdrnr_config.gadget = GadgetType::kNone;
    cdrnr_config.opponent_model = &uniform_tabular;
    cdrnr_config.p = 1.0;
    cdrnr_config.target_player = target;
    cdrnr_config.cfr_iterations = iters;

    auto cdrnr_result = ContinualResolve(
        game, uniform, cdrnr_config, depth, MVSGame::DepthMode::kActionBased);
    double cdrnr_gain = GainAgainst(
        *game, target, *cdrnr_result, uniform_tabular);
    double cdrnr_exp = algorithms::Exploitability(*game, *cdrnr_result);

    std::cout << "  [P" << target << "] BR:    gain=" << br_gain << std::endl;
    std::cout << "  [P" << target << "] CDRNR: gain=" << cdrnr_gain
              << ", expl=" << cdrnr_exp << std::endl;
    std::cout << "  [P" << target << "] Gain diff: "
              << std::abs(cdrnr_gain - br_gain) << std::endl;

    // CDRNR p=1 should exploit uniform similarly to full BR
    SPIEL_CHECK_GT(cdrnr_gain, br_gain - 0.2);
  }

  std::cout << "TestGoofspiel_p1_vs_BR PASSED" << std::endl;
}

// =============================================================================
// Test 15: Leduc p=0 vs Gadget resolve (should be equivalent)
// =============================================================================

void TestLeduc_p0_vs_Gadget() {
  std::cout << "TestLeduc_p0_vs_Gadget..." << std::endl;

  auto game = LoadGame("leduc_poker");
  const int iters = 500;
  const int depth = 3;

  // --- Baseline: standard gadget resolve (CFR on full game + gadget) ---
  algorithms::CFRSolverBase cfr_solver(*game, true, true, true);
  for (int i = 0; i < iters; ++i) cfr_solver.EvaluateAndUpdatePolicy();
  auto cfr_policy = cfr_solver.AveragePolicy();
  TabularPolicy cfr_trunk = cfr_solver.TabularAveragePolicy();

  auto decomp = DecomposeGameAtRound(game, *cfr_policy, depth);

  ResolvingConfig gadget_config;
  gadget_config.solver = SolverType::kCFR;
  gadget_config.gadget = GadgetType::kResolving;
  gadget_config.cfr_iterations = iters;

  auto gadget_result = ResolveSubgames(decomp, cfr_trunk, gadget_config);
  double gadget_exp = algorithms::Exploitability(*game, *gadget_result);

  UniformPolicy uniform;
  TabularPolicy uniform_tabular = MakeUniformTabular(*game);

  // --- Test CDRNR p=0 for both players ---
  for (int target = 0; target < 2; ++target) {
    double gadget_gain = GainAgainst(
        *game, target, *gadget_result, uniform_tabular);

    ResolvingConfig cdrnr_config;
    cdrnr_config.solver = SolverType::kRNR;
    cdrnr_config.gadget = GadgetType::kResolving;
    cdrnr_config.opponent_model = &uniform_tabular;
    cdrnr_config.p = 0.0;
    cdrnr_config.target_player = target;
    cdrnr_config.cfr_iterations = iters;

    auto cdrnr_result = ContinualResolve(
        game, uniform, cdrnr_config, depth, MVSGame::DepthMode::kRoundBased);
    double cdrnr_exp = algorithms::Exploitability(*game, *cdrnr_result);
    double cdrnr_gain = GainAgainst(
        *game, target, *cdrnr_result, uniform_tabular);

    std::cout << "  [P" << target << "] Gadget: expl=" << gadget_exp
              << ", gain=" << gadget_gain << std::endl;
    std::cout << "  [P" << target << "] CDRNR:  expl=" << cdrnr_exp
              << ", gain=" << cdrnr_gain << std::endl;

    // CDRNR p=0 should produce similar exploitability as gadget resolve
    SPIEL_CHECK_LT(cdrnr_exp, 0.1);
    SPIEL_CHECK_LT(std::abs(cdrnr_exp - gadget_exp), 0.05);
  }

  std::cout << "TestLeduc_p0_vs_Gadget PASSED" << std::endl;
}

// =============================================================================
// Test 16: Leduc p=1 vs best response (should be equivalent)
// =============================================================================

void TestLeduc_p1_vs_BR() {
  std::cout << "TestLeduc_p1_vs_BR..." << std::endl;

  auto game = LoadGame("leduc_poker");
  UniformPolicy uniform;
  TabularPolicy uniform_tabular = MakeUniformTabular(*game);
  const int iters = 500;
  const int depth = 3;

  for (int target = 0; target < 2; ++target) {
    // --- Baseline: tabular best response against uniform ---
    algorithms::TabularBestResponse br(*game, target, &uniform_tabular);
    TabularPolicy br_policy = br.GetBestResponsePolicy();
    double br_gain = GainAgainst(*game, target, br_policy, uniform_tabular);

    // --- CDRNR with p=1 (should be close to BR) ---
    ResolvingConfig cdrnr_config;
    cdrnr_config.solver = SolverType::kRNR;
    cdrnr_config.gadget = GadgetType::kNone;
    cdrnr_config.opponent_model = &uniform_tabular;
    cdrnr_config.p = 1.0;
    cdrnr_config.target_player = target;
    cdrnr_config.cfr_iterations = iters;

    auto cdrnr_result = ContinualResolve(
        game, uniform, cdrnr_config, depth, MVSGame::DepthMode::kRoundBased);
    double cdrnr_gain = GainAgainst(
        *game, target, *cdrnr_result, uniform_tabular);
    double cdrnr_exp = algorithms::Exploitability(*game, *cdrnr_result);

    std::cout << "  [P" << target << "] BR:    gain=" << br_gain << std::endl;
    std::cout << "  [P" << target << "] CDRNR: gain=" << cdrnr_gain
              << ", expl=" << cdrnr_exp << std::endl;
    std::cout << "  [P" << target << "] Gain diff: "
              << std::abs(cdrnr_gain - br_gain) << std::endl;

    // CDRNR p=1 should exploit uniform similarly to full BR
    SPIEL_CHECK_GT(cdrnr_gain, br_gain - 0.5);
  }

  std::cout << "TestLeduc_p1_vs_BR PASSED" << std::endl;
}

// =============================================================================
// Main
// =============================================================================

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  std::string arg = (argc > 1) ? std::string(argv[1]) : "";

  if (arg == "only_sweep") {
    open_spiel::TestSweepP();
    return 0;
  }
  if (arg == "goofspiel") {
    open_spiel::TestCDRNRGoofspiel();
    open_spiel::TestGoofspiel_p0_vs_Gadget();
    open_spiel::TestGoofspiel_p1_vs_BR();
    return 0;
  }
  if (arg == "leduc") {
    open_spiel::TestCDRNRLeduc();
    open_spiel::TestLeduc_p0_vs_Gadget();
    open_spiel::TestLeduc_p1_vs_BR();
    return 0;
  }

  open_spiel::TestGadgetPolicyWrapper();
  open_spiel::TestResolveSubgamesCFR();
  open_spiel::TestCDBRKuhn();
  open_spiel::TestCDRNRKuhnResolving();
  open_spiel::TestCDRNRKuhnMaxMargin();
  open_spiel::TestCDRNRLeducResolving();
  open_spiel::TestCDRNRLeducMaxMargin();
  open_spiel::TestSweepP();
  open_spiel::TestCDRNR_p0_vs_Gadget();
  open_spiel::TestCDRNR_p1_vs_BR();
  open_spiel::TestCDRNRGoofspiel();
  open_spiel::TestCDRNRLeduc();
  open_spiel::TestGoofspiel_p0_vs_Gadget();
  open_spiel::TestGoofspiel_p1_vs_BR();
  open_spiel::TestLeduc_p0_vs_Gadget();
  open_spiel::TestLeduc_p1_vs_BR();

  std::cout << "\nAll tests passed!" << std::endl;
  return 0;
}
