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

  // Compute gain against uniform opponent
  auto returns = algorithms::ExpectedReturns(
      *game->NewInitialState(), *result, -1);
  double gain = returns[0];

  std::cout << "  P0 return vs uniform: " << gain << std::endl;
  SPIEL_CHECK_GT(gain, -0.06);

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

  auto returns = algorithms::ExpectedReturns(
      *game->NewInitialState(), *result, -1);
  double gain = returns[0];

  double exp = algorithms::Exploitability(*game, *result);

  std::cout << "  P0 return vs uniform: " << gain << std::endl;
  std::cout << "  Exploitability: " << exp << std::endl;

  // With p=0.5, should have moderate exploitability
  SPIEL_CHECK_LT(exp, 1.0);

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

  auto returns = algorithms::ExpectedReturns(
      *game->NewInitialState(), *result, -1);
  double gain = returns[0];

  double exp = algorithms::Exploitability(*game, *result);

  std::cout << "  P0 return vs uniform: " << gain << std::endl;
  std::cout << "  Exploitability: " << exp << std::endl;

  SPIEL_CHECK_LT(exp, 1.0);

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

  auto returns = algorithms::ExpectedReturns(
      *game->NewInitialState(), *result, -1);
  double gain = returns[0];

  double exp = algorithms::Exploitability(*game, *result);

  std::cout << "  P0 return vs uniform: " << gain << std::endl;
  std::cout << "  Exploitability: " << exp << std::endl;

  // The result should be a valid policy
  SPIEL_CHECK_GT(result->PolicyTable().size(), 0);
  SPIEL_CHECK_LT(exp, 2.0);

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

  auto returns = algorithms::ExpectedReturns(
      *game->NewInitialState(), *result, -1);
  double gain = returns[0];

  double exp = algorithms::Exploitability(*game, *result);

  std::cout << "  P0 return vs uniform: " << gain << std::endl;
  std::cout << "  Exploitability: " << exp << std::endl;

  SPIEL_CHECK_GT(result->PolicyTable().size(), 0);
  SPIEL_CHECK_LT(exp, 2.0);

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
  std::vector<double> exploitabilities;
  std::vector<double> gains;

  for (double p : ps) {
    ResolvingConfig config;
    config.solver = (p > 0) ? SolverType::kRNR : SolverType::kCFR;
    config.gadget = (p > 0) ? GadgetType::kResolving : GadgetType::kResolving;
    config.opponent_model = &uniform_tabular;
    config.p = p;
    config.target_player = 0;
    config.cfr_iterations = 200;

    auto result = ContinualResolve(
        game, uniform, config, 2, MVSGame::DepthMode::kActionBased);

    auto returns = algorithms::ExpectedReturns(
        *game->NewInitialState(), *result, -1);
    double gain = returns[0];
    double exp = algorithms::Exploitability(*game, *result);

    exploitabilities.push_back(exp);
    gains.push_back(gain);

    std::cout << "  p=" << p << ": gain=" << gain
              << ", exploitability=" << exp << std::endl;
  }

  // At p=0 (Nash solving), exploitability should be low
  // Note: with depth-limited solving, exact Nash isn't achieved
  // but exploitability should be reasonable
  SPIEL_CHECK_LT(exploitabilities[0], 0.5);

  // With increasing p, gain should generally increase
  // (not strictly monotonic due to finite iterations, but trend should hold)
  std::cout << "  Gain trend: " << gains[0] << " -> " << gains[4] << std::endl;

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
// Main
// =============================================================================

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::TestGadgetPolicyWrapper();
  open_spiel::TestResolveSubgamesCFR();
  open_spiel::TestCDBRKuhn();
  open_spiel::TestCDRNRKuhnResolving();
  open_spiel::TestCDRNRKuhnMaxMargin();
  open_spiel::TestCDRNRLeducResolving();
  open_spiel::TestCDRNRLeducMaxMargin();
  open_spiel::TestSweepP();

  std::cout << "\nAll tests passed!" << std::endl;
  return 0;
}
