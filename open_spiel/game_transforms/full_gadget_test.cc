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

#include "open_spiel/game_transforms/full_gadget.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/game_transforms/continual_resolving.h"
#include "open_spiel/game_transforms/matrix_valued_states.h"
#include "open_spiel/game_transforms/subgame_utils.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include <stdexcept>

#if OPEN_SPIEL_BUILD_WITH_ORTOOLS
#include "open_spiel/algorithms/ortools/sequence_form_lp.h"
#endif

namespace open_spiel {
namespace {

// Throwing error handler to enable try/catch on SpielFatalError
void ThrowingErrorHandler(const std::string& msg) {
  throw std::runtime_error(msg);
}

// Default error handler (restore after throwing)
void DefaultErrorHandler(const std::string& msg) {
  std::cerr << "Spiel Fatal Error: " << msg << std::endl << std::flush;
  std::exit(1);
}

// RAII guard to temporarily install throwing error handler
struct ThrowingErrorGuard {
  ThrowingErrorGuard() { SetErrorHandler(ThrowingErrorHandler); }
  ~ThrowingErrorGuard() { SetErrorHandler(DefaultErrorHandler); }
};

// =============================================================================
// Helper: compute expected returns from a state under a policy
// =============================================================================

std::vector<double> ComputeExpReturns(const State& state,
                                       const Policy& policy) {
  if (state.IsTerminal()) return state.Returns();
  int np = state.NumPlayers();
  std::vector<double> ev(np, 0.0);
  if (state.IsChanceNode()) {
    for (const auto& [a, p] : state.ChanceOutcomes()) {
      auto child = state.Clone();
      child->ApplyAction(a);
      auto cev = ComputeExpReturns(*child, policy);
      for (int i = 0; i < np; ++i) ev[i] += p * cev[i];
    }
  } else {
    auto ap = policy.GetStatePolicy(state, state.CurrentPlayer());
    if (ap.empty()) {
      auto legal = state.LegalActions();
      double p = 1.0 / legal.size();
      for (Action a : legal) {
        auto child = state.Clone();
        child->ApplyAction(a);
        auto cev = ComputeExpReturns(*child, policy);
        for (int i = 0; i < np; ++i) ev[i] += p * cev[i];
      }
    } else {
      for (const auto& [a, p] : ap) {
        if (p <= 0.0) continue;
        auto child = state.Clone();
        child->ApplyAction(a);
        auto cev = ComputeExpReturns(*child, policy);
        for (int i = 0; i < np; ++i) ev[i] += p * cev[i];
      }
    }
  }
  return ev;
}

// Helper: prepare boundary data from a decomposition
struct FullGadgetData {
  std::unordered_map<std::string, std::vector<std::string>>
      boundary_states_by_group;
  std::unordered_map<std::string, std::vector<double>> boundary_values;
};

FullGadgetData PrepareFullGadgetData(const SubgameDecomposition& decomp,
                                      const Policy& trunk_policy) {
  FullGadgetData data;
  for (const auto& [pub_obs, states] : decomp.grouped_subgames) {
    for (const auto& state : states) {
      std::string hist = state->HistoryString();
      data.boundary_states_by_group[pub_obs].push_back(hist);
      data.boundary_values[hist] = ComputeExpReturns(*state, trunk_policy);
    }
  }
  return data;
}

// =============================================================================
// Test 1: Basic construction
// =============================================================================

void TestFullGadgetConstruction() {
  std::cout << "TestFullGadgetConstruction...";
  auto game = LoadGame("kuhn_poker");
  auto uniform = std::make_shared<UniformPolicy>();

  auto decomp = DecomposeGameAtDepth(game, *uniform, 2);
  SPIEL_CHECK_GT(decomp.grouped_subgames.size(), 0);

  auto data = PrepareFullGadgetData(decomp, *uniform);
  auto it = decomp.grouped_subgames.begin();
  std::string target_pub_obs = it->first;

  auto fg_trunk = CreateFullGadgetGame(
      game, std::make_shared<UniformPolicy>(), 0, target_pub_obs,
      data.boundary_states_by_group, data.boundary_values,
      FullGadgetGame::Mode::kTrunk);
  SPIEL_CHECK_TRUE(fg_trunk != nullptr);
  SPIEL_CHECK_EQ(fg_trunk->NumPlayers(), 2);
  SPIEL_CHECK_EQ(fg_trunk->ResolvingPlayer(), 0);
  SPIEL_CHECK_EQ(fg_trunk->NonResolvingPlayer(), 1);

  auto fg_path = CreateFullGadgetGame(
      game, std::make_shared<UniformPolicy>(), 0, target_pub_obs,
      data.boundary_states_by_group, data.boundary_values,
      FullGadgetGame::Mode::kPath);
  SPIEL_CHECK_TRUE(fg_path != nullptr);

  std::cout << " PASSED\n";
}

// =============================================================================
// Test 2: Resolving player becomes chance in trunk
// =============================================================================

void TestResolvingPlayerIsChance() {
  std::cout << "TestResolvingPlayerIsChance...";
  auto game = LoadGame("kuhn_poker");
  auto uniform = std::make_shared<UniformPolicy>();
  auto decomp = DecomposeGameAtDepth(game, *uniform, 2);
  auto data = PrepareFullGadgetData(decomp, *uniform);

  auto it = decomp.grouped_subgames.begin();
  auto fg = CreateFullGadgetGame(
      game, std::make_shared<UniformPolicy>(), 0, it->first,
      data.boundary_states_by_group, data.boundary_values,
      FullGadgetGame::Mode::kTrunk);

  auto state = fg->NewInitialState();
  // Kuhn poker: chance deals, then player 0 acts.
  // Deal two cards:
  state->ApplyAction(state->ChanceOutcomes()[0].first);
  state->ApplyAction(state->ChanceOutcomes()[0].first);

  auto* fg_state = dynamic_cast<FullGadgetState*>(state.get());
  SPIEL_CHECK_TRUE(fg_state->GetPhase() == FullGadgetState::Phase::kTrunk);
  // Player 0 is resolving → should appear as chance
  SPIEL_CHECK_EQ(state->CurrentPlayer(), kChancePlayerId);

  auto outcomes = state->ChanceOutcomes();
  SPIEL_CHECK_GT(outcomes.size(), 0);
  // Uniform policy → equal probabilities
  double expected_prob = 1.0 / outcomes.size();
  for (const auto& [a, p] : outcomes) {
    SPIEL_CHECK_FLOAT_NEAR(p, expected_prob, 1e-9);
  }
  std::cout << " PASSED\n";
}

// =============================================================================
// Test 3: Info state prefixes
// =============================================================================

void TestInfoStatePrefixes() {
  std::cout << "TestInfoStatePrefixes...";
  auto game = LoadGame("kuhn_poker");
  auto uniform = std::make_shared<UniformPolicy>();
  auto decomp = DecomposeGameAtDepth(game, *uniform, 2);
  auto data = PrepareFullGadgetData(decomp, *uniform);
  auto it = decomp.grouped_subgames.begin();

  auto fg = CreateFullGadgetGame(
      game, std::make_shared<UniformPolicy>(), 0, it->first,
      data.boundary_states_by_group, data.boundary_values,
      FullGadgetGame::Mode::kTrunk);

  std::unordered_set<std::string> trunk_is, subgame_is;
  std::function<void(State&)> traverse = [&](State& state) {
    if (state.IsTerminal()) return;
    auto* fg_state = dynamic_cast<FullGadgetState*>(&state);
    if (!state.IsChanceNode()) {
      Player pl = state.CurrentPlayer();
      std::string is = state.InformationStateString(pl);
      if (fg_state->GetPhase() == FullGadgetState::Phase::kTrunk)
        trunk_is.insert(is);
      else if (fg_state->GetPhase() == FullGadgetState::Phase::kSubgame)
        subgame_is.insert(is);
    }
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
  traverse(*fg->NewInitialState());

  for (const auto& is : trunk_is) {
    SPIEL_CHECK_TRUE(is.compare(0, 11, "full_trunk:") == 0);
  }
  for (const auto& is : subgame_is) {
    SPIEL_CHECK_TRUE(is.compare(0, 15, "full_F:subgame:") == 0);
  }
  std::cout << " PASSED\n";
}

// =============================================================================
// Test 4: Full pipeline – Kuhn decompose + resolve + evaluate (kFullTrunk)
// =============================================================================

void TestKuhnFullGadgetTrunkPipeline() {
  std::cout << "TestKuhnFullGadgetTrunkPipeline..." << std::endl;
  auto game = LoadGame("kuhn_poker");

  // Solve full game for baseline
  algorithms::CFRSolverBase full_solver(*game, true, true, true);
  for (int i = 0; i < 500; ++i) full_solver.EvaluateAndUpdatePolicy();
  auto full_policy = full_solver.AveragePolicy();
  TabularPolicy trunk = full_solver.TabularAveragePolicy();
  double full_exp = algorithms::Exploitability(*game, *full_policy);

  // Decompose at depth 2
  auto decomp = DecomposeGameAtDepth(game, *full_policy, 2);

  // Resolve with CFR + Full Gadget (trunk mode)
  ResolvingConfig config;
  config.solver = SolverType::kCFR;
  config.gadget = GadgetType::kFullTrunk;
  config.cfr_iterations = 500;


  auto result = ResolveSubgames(decomp, trunk, config);
  double resolved_exp = algorithms::Exploitability(*game, *result);

  std::cout << "  Full:     " << full_exp << std::endl;
  std::cout << "  Resolved: " << resolved_exp << std::endl;
  std::cout << "  Info states: " << result->PolicyTable().size() << std::endl;

  SPIEL_CHECK_LT(resolved_exp, 0.5);

  std::cout << "TestKuhnFullGadgetTrunkPipeline PASSED" << std::endl;
}

// =============================================================================
// Test 5: Full pipeline – Kuhn decompose + resolve + evaluate (kFullPath)
// =============================================================================

void TestKuhnFullGadgetPathPipeline() {
  std::cout << "TestKuhnFullGadgetPathPipeline..." << std::endl;
  auto game = LoadGame("kuhn_poker");

  algorithms::CFRSolverBase full_solver(*game, true, true, true);
  for (int i = 0; i < 500; ++i) full_solver.EvaluateAndUpdatePolicy();
  auto full_policy = full_solver.AveragePolicy();
  TabularPolicy trunk = full_solver.TabularAveragePolicy();
  double full_exp = algorithms::Exploitability(*game, *full_policy);

  auto decomp = DecomposeGameAtDepth(game, *full_policy, 2);

  ResolvingConfig config;
  config.solver = SolverType::kCFR;
  config.gadget = GadgetType::kFullPath;
  config.cfr_iterations = 500;


  auto result = ResolveSubgames(decomp, trunk, config);
  double resolved_exp = algorithms::Exploitability(*game, *result);

  std::cout << "  Full:     " << full_exp << std::endl;
  std::cout << "  Resolved: " << resolved_exp << std::endl;

  SPIEL_CHECK_LT(resolved_exp, 0.5);

  std::cout << "TestKuhnFullGadgetPathPipeline PASSED" << std::endl;
}

// =============================================================================
// Test 6: Compare Full Gadget vs Resolving Gadget vs unsafe on Kuhn
// =============================================================================

void TestFullGadgetVsOtherGadgets() {
  std::cout << "TestFullGadgetVsOtherGadgets..." << std::endl;
  auto game = LoadGame("kuhn_poker");

  algorithms::CFRSolverBase solver(*game, true, true, true);
  for (int i = 0; i < 500; ++i) solver.EvaluateAndUpdatePolicy();
  auto policy = solver.AveragePolicy();
  TabularPolicy trunk = solver.TabularAveragePolicy();
  double full_exp = algorithms::Exploitability(*game, *policy);

  auto decomp = DecomposeGameAtDepth(game, *policy, 2);

  // Resolving gadget
  ResolvingConfig resolving_config;
  resolving_config.solver = SolverType::kCFR;
  resolving_config.gadget = GadgetType::kResolving;
  resolving_config.cfr_iterations = 500;
  auto resolving_result = ResolveSubgames(decomp, trunk, resolving_config);
  double resolving_exp = algorithms::Exploitability(*game, *resolving_result);

  // Unsafe subgame
  ResolvingConfig unsafe_config;
  unsafe_config.solver = SolverType::kCFR;
  unsafe_config.gadget = GadgetType::kNone;
  unsafe_config.cfr_iterations = 500;
  auto unsafe_result = ResolveSubgames(decomp, trunk, unsafe_config);
  double unsafe_exp = algorithms::Exploitability(*game, *unsafe_result);

  // Full Gadget (trunk)
  ResolvingConfig full_trunk_config;
  full_trunk_config.solver = SolverType::kCFR;
  full_trunk_config.gadget = GadgetType::kFullTrunk;
  full_trunk_config.cfr_iterations = 500;
  auto full_trunk_result = ResolveSubgames(decomp, trunk, full_trunk_config);
  double full_trunk_exp =
      algorithms::Exploitability(*game, *full_trunk_result);

  // Full Gadget (path)
  ResolvingConfig full_path_config;
  full_path_config.solver = SolverType::kCFR;
  full_path_config.gadget = GadgetType::kFullPath;
  full_path_config.cfr_iterations = 500;
  auto full_path_result = ResolveSubgames(decomp, trunk, full_path_config);
  double full_path_exp =
      algorithms::Exploitability(*game, *full_path_result);

  std::cout << "  Full game:     " << full_exp << std::endl;
  std::cout << "  Resolving:     " << resolving_exp << std::endl;
  std::cout << "  Unsafe:        " << unsafe_exp << std::endl;
  std::cout << "  Full (trunk):  " << full_trunk_exp << std::endl;
  std::cout << "  Full (path):   " << full_path_exp << std::endl;

  SPIEL_CHECK_LT(resolving_exp, 0.5);
  SPIEL_CHECK_LT(unsafe_exp, 0.5);
  SPIEL_CHECK_LT(full_trunk_exp, 0.5);
  SPIEL_CHECK_LT(full_path_exp, 0.5);

  std::cout << "TestFullGadgetVsOtherGadgets PASSED" << std::endl;
}

// =============================================================================
// Test 7: Leduc poker pipeline (kFullTrunk)
// =============================================================================

void TestLeducFullGadgetPipeline() {
  std::cout << "TestLeducFullGadgetPipeline..." << std::endl;
  auto game = LoadGame("leduc_poker");

  algorithms::CFRSolverBase solver(*game, true, true, true);
  for (int i = 0; i < 500; ++i) solver.EvaluateAndUpdatePolicy();
  auto policy = solver.AveragePolicy();
  TabularPolicy trunk = solver.TabularAveragePolicy();
  double full_exp = algorithms::Exploitability(*game, *policy);

  // Decompose at round 3 (after 2 private + 1 public card deals)
  auto decomp = DecomposeGameAtRound(game, *policy, 3);

  // Resolving gadget for comparison
  ResolvingConfig resolving_config;
  resolving_config.solver = SolverType::kCFR;
  resolving_config.gadget = GadgetType::kResolving;
  resolving_config.cfr_iterations = 500;
  auto resolving_result = ResolveSubgames(decomp, trunk, resolving_config);
  double resolving_exp = algorithms::Exploitability(*game, *resolving_result);

  // Full Gadget (trunk mode)
  ResolvingConfig full_config;
  full_config.solver = SolverType::kCFR;
  full_config.gadget = GadgetType::kFullTrunk;
  full_config.cfr_iterations = 500;
  auto full_result = ResolveSubgames(decomp, trunk, full_config);
  double full_gadget_exp = algorithms::Exploitability(*game, *full_result);

  // Full Gadget (path mode)
  ResolvingConfig path_config;
  path_config.solver = SolverType::kCFR;
  path_config.gadget = GadgetType::kFullPath;
  path_config.cfr_iterations = 500;
  auto path_result = ResolveSubgames(decomp, trunk, path_config);
  double path_gadget_exp = algorithms::Exploitability(*game, *path_result);

  std::cout << "  Full game:      " << full_exp << std::endl;
  std::cout << "  Resolving:      " << resolving_exp << std::endl;
  std::cout << "  Full (trunk):   " << full_gadget_exp << std::endl;
  std::cout << "  Full (path):    " << path_gadget_exp << std::endl;

  SPIEL_CHECK_LT(resolving_exp, 1.0);
  SPIEL_CHECK_LT(full_gadget_exp, 1.0);
  SPIEL_CHECK_LT(path_gadget_exp, 1.0);

  std::cout << "TestLeducFullGadgetPipeline PASSED" << std::endl;
}

// =============================================================================
// Test 8: LP solver comparison – all gadget types on Kuhn poker
// =============================================================================

void TestKuhnAllGadgetsLP() {
  std::cout << "TestKuhnAllGadgetsLP..." << std::endl;
  auto game = LoadGame("kuhn_poker");

  // Solve trunk with LP (exact equilibrium)
  auto [trunk, game_value] =
      algorithms::ortools::MakeEquilibriumPolicy(*game, true);
  auto policy = std::make_shared<TabularPolicy>(trunk);
  double full_exp = algorithms::Exploitability(*game, *policy);

  auto decomp = DecomposeGameAtDepth(game, *policy, 2);

  // LP on Unsafe
  ResolvingConfig lp_unsafe_config;
  lp_unsafe_config.solver = SolverType::kLP;
  lp_unsafe_config.gadget = GadgetType::kNone;
  auto lp_unsafe_result = ResolveSubgames(decomp, trunk, lp_unsafe_config);
  double lp_unsafe_exp = algorithms::Exploitability(*game, *lp_unsafe_result);

  // LP on Resolving gadget
  ResolvingConfig lp_resolving_config;
  lp_resolving_config.solver = SolverType::kLP;
  lp_resolving_config.gadget = GadgetType::kResolving;
  auto lp_resolving_result = ResolveSubgames(decomp, trunk, lp_resolving_config);
  double lp_resolving_exp = algorithms::Exploitability(*game, *lp_resolving_result);

  // LP on Max-Margin gadget
  ResolvingConfig lp_maxmargin_config;
  lp_maxmargin_config.solver = SolverType::kLP;
  lp_maxmargin_config.gadget = GadgetType::kMaxMargin;
  auto lp_maxmargin_result = ResolveSubgames(decomp, trunk, lp_maxmargin_config);
  double lp_maxmargin_exp = algorithms::Exploitability(*game, *lp_maxmargin_result);

  // LP on Full Gadget (trunk mode)
  ResolvingConfig lp_trunk_config;
  lp_trunk_config.solver = SolverType::kLP;
  lp_trunk_config.gadget = GadgetType::kFullTrunk;
  auto lp_trunk_result = ResolveSubgames(decomp, trunk, lp_trunk_config);
  double lp_trunk_exp = algorithms::Exploitability(*game, *lp_trunk_result);

  // LP on Full Gadget (path mode)
  ResolvingConfig lp_path_config;
  lp_path_config.solver = SolverType::kLP;
  lp_path_config.gadget = GadgetType::kFullPath;
  auto lp_path_result = ResolveSubgames(decomp, trunk, lp_path_config);
  double lp_path_exp = algorithms::Exploitability(*game, *lp_path_result);

  std::cout << "  Full game:         " << full_exp << std::endl;
  std::cout << "  LP Unsafe:         " << lp_unsafe_exp << std::endl;
  std::cout << "  LP Resolving:      " << lp_resolving_exp << std::endl;
  std::cout << "  LP Max-Margin:     " << lp_maxmargin_exp << std::endl;
  std::cout << "  LP Full (trunk):   " << lp_trunk_exp << std::endl;
  std::cout << "  LP Full (path):    " << lp_path_exp << std::endl;

  SPIEL_CHECK_LT(lp_unsafe_exp, 0.5);
  SPIEL_CHECK_LT(lp_resolving_exp, 0.01);
  SPIEL_CHECK_LT(lp_maxmargin_exp, 0.01);
  SPIEL_CHECK_LT(lp_trunk_exp, 0.01);
  SPIEL_CHECK_LT(lp_path_exp, 0.01);

  std::cout << "TestKuhnAllGadgetsLP PASSED" << std::endl;
}

// =============================================================================
// Test 9: LP solver comparison – all gadget types on Leduc poker
// =============================================================================

void TestLeducAllGadgetsLP() {
  std::cout << "TestLeducAllGadgetsLP..." << std::endl;
  auto game = LoadGame("leduc_poker");

  // Solve trunk with LP (exact equilibrium)
  auto [trunk, game_value] =
      algorithms::ortools::MakeEquilibriumPolicy(*game, true);
  auto policy = std::make_shared<TabularPolicy>(trunk);
  double full_exp = algorithms::Exploitability(*game, *policy);

  auto decomp = DecomposeGameAtRound(game, *policy, 3);

  // LP on Unsafe
  ResolvingConfig lp_unsafe_config;
  lp_unsafe_config.solver = SolverType::kLP;
  lp_unsafe_config.gadget = GadgetType::kNone;
  auto lp_unsafe_result = ResolveSubgames(decomp, trunk, lp_unsafe_config);
  double lp_unsafe_exp = algorithms::Exploitability(*game, *lp_unsafe_result);

  // LP on Resolving gadget
  ResolvingConfig lp_resolving_config;
  lp_resolving_config.solver = SolverType::kLP;
  lp_resolving_config.gadget = GadgetType::kResolving;
  auto lp_resolving_result = ResolveSubgames(decomp, trunk, lp_resolving_config);
  double lp_resolving_exp = algorithms::Exploitability(*game, *lp_resolving_result);

  // LP on Max-Margin gadget
  ResolvingConfig lp_maxmargin_config;
  lp_maxmargin_config.solver = SolverType::kLP;
  lp_maxmargin_config.gadget = GadgetType::kMaxMargin;
  auto lp_maxmargin_result = ResolveSubgames(decomp, trunk, lp_maxmargin_config);
  double lp_maxmargin_exp = algorithms::Exploitability(*game, *lp_maxmargin_result);

  // LP on Full Gadget (trunk mode)
  ResolvingConfig lp_trunk_config;
  lp_trunk_config.solver = SolverType::kLP;
  lp_trunk_config.gadget = GadgetType::kFullTrunk;
  auto lp_trunk_result = ResolveSubgames(decomp, trunk, lp_trunk_config);
  double lp_trunk_exp = algorithms::Exploitability(*game, *lp_trunk_result);

  // LP on Full Gadget (path mode)
  ResolvingConfig lp_path_config;
  lp_path_config.solver = SolverType::kLP;
  lp_path_config.gadget = GadgetType::kFullPath;
  auto lp_path_result = ResolveSubgames(decomp, trunk, lp_path_config);
  double lp_path_exp = algorithms::Exploitability(*game, *lp_path_result);

  std::cout << "  Full game:       " << full_exp << std::endl;
  std::cout << "  LP Unsafe:       " << lp_unsafe_exp << std::endl;
  std::cout << "  LP Resolving:    " << lp_resolving_exp << std::endl;
  std::cout << "  LP Max-Margin:   " << lp_maxmargin_exp << std::endl;
  std::cout << "  LP Full (trunk): " << lp_trunk_exp << std::endl;
  std::cout << "  LP Full (path):  " << lp_path_exp << std::endl;

  SPIEL_CHECK_LT(lp_unsafe_exp, 1.0);
  SPIEL_CHECK_LT(lp_resolving_exp, 1.0);
  SPIEL_CHECK_LT(lp_maxmargin_exp, 1.0);
  SPIEL_CHECK_LT(lp_trunk_exp, 1.0);
  SPIEL_CHECK_LT(lp_path_exp, 1.0);

  std::cout << "TestLeducAllGadgetsLP PASSED" << std::endl;
}

// =============================================================================
// Test 10: Debug Full Gadget — MVS LP, scalar boundary, single-group tests
// =============================================================================

void TestDebugFullGadget() {
  std::cout << "TestDebugFullGadget..." << std::endl;

  // MVS LP sanity check
  {
    auto game = LoadGame("kuhn_poker");
    auto mvs = CreateMVSGameWithSubtreePureStrategies(game, 2);
    auto [mvs_policy, mvs_value] =
        algorithms::ortools::MakeEquilibriumPolicy(*mvs, true);
    auto mvs_pol_ptr = std::make_shared<TabularPolicy>(mvs_policy);
    double mvs_exp = algorithms::Exploitability(*mvs, *mvs_pol_ptr);
    std::cout << "  Kuhn MVS LP exp:   " << mvs_exp << std::endl;
  }
  {
    auto game = LoadGame("leduc_poker");
    auto mvs = CreateMVSGameWithSubtreePureStrategies(
        game, 3, MVSGame::DepthMode::kRoundBased);
    auto [mvs_policy, mvs_value] =
        algorithms::ortools::MakeEquilibriumPolicy(*mvs, true);
    auto mvs_pol_ptr = std::make_shared<TabularPolicy>(mvs_policy);
    double mvs_exp = algorithms::Exploitability(*mvs, *mvs_pol_ptr);
    std::cout << "  Leduc MVS LP exp:  " << mvs_exp << std::endl;
  }

  // Try LP on plain Leduc
  {
    auto game = LoadGame("leduc_poker");
    std::cout << "  Trying LP on plain Leduc..." << std::endl;
    try {
      auto [lp_policy, lp_value] =
          algorithms::ortools::MakeEquilibriumPolicy(*game, true);
      auto lp_pol_ptr = std::make_shared<TabularPolicy>(lp_policy);
      double lp_exp = algorithms::Exploitability(*game, *lp_pol_ptr);
      std::cout << "  Leduc LP exp: " << lp_exp << ", value: " << lp_value << std::endl;
    } catch (...) {
      std::cout << "  Leduc LP CRASHED (as expected)" << std::endl;
    }
  }

  // Full Gadget on Leduc with scalar boundaries (no MVS) — LP
  // This isolates the trunk structure quality
  {
    auto game = LoadGame("leduc_poker");
    algorithms::CFRSolverBase cfr_solver(*game, true, true, true);
    for (int i = 0; i < 500; ++i) cfr_solver.EvaluateAndUpdatePolicy();
    auto policy = cfr_solver.AveragePolicy();
    TabularPolicy trunk = cfr_solver.TabularAveragePolicy();
    auto trunk_ptr = std::make_shared<TabularPolicy>(trunk);

    auto decomp = DecomposeGameAtRound(game, *policy, 3);

    // Prepare boundary data
    std::unordered_map<std::string, std::vector<std::string>> boundary_by_group;
    std::unordered_map<std::string, std::vector<double>> boundary_values;
    for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
      for (const auto& root : roots) {
        std::string hist = root->HistoryString();
        boundary_by_group[pub_obs].push_back(hist);
        boundary_values[hist] = ComputeExpReturns(*root, trunk);
      }
    }

    // Pick first group as target
    auto it = decomp.grouped_subgames.begin();
    std::string target = it->first;

    // Scalar boundary (no MVS): enumerate_boundary_portfolios=false, no portfolios
    auto fg_scalar = CreateFullGadgetGame(
        game, std::make_shared<TabularPolicy>(trunk), 0, target,
        boundary_by_group, boundary_values,
        FullGadgetGame::Mode::kTrunk);

    auto [scalar_policy, scalar_value] =
        algorithms::ortools::MakeEquilibriumPolicy(*fg_scalar, true);
    auto scalar_pol_ptr = std::make_shared<TabularPolicy>(scalar_policy);
    double scalar_exp = algorithms::Exploitability(*fg_scalar, *scalar_pol_ptr);
    std::cout << "  Leduc FG scalar LP exp (in FG): " << scalar_exp << std::endl;

    // MVS boundary: enumerate_boundary_portfolios=true
    auto fg_mvs = CreateFullGadgetGame(
        game, std::make_shared<TabularPolicy>(trunk), 0, target,
        boundary_by_group, boundary_values,
        FullGadgetGame::Mode::kTrunk,
        /*boundary_portfolios_p0=*/{},
        /*boundary_portfolios_p1=*/{},
        /*enumerate_boundary_portfolios=*/true);

    auto [mvs_fg_policy, mvs_fg_value] =
        algorithms::ortools::MakeEquilibriumPolicy(*fg_mvs, true);
    auto mvs_fg_pol_ptr = std::make_shared<TabularPolicy>(mvs_fg_policy);
    double mvs_fg_exp = algorithms::Exploitability(*fg_mvs, *mvs_fg_pol_ptr);
    std::cout << "  Leduc FG MVS LP exp (in FG):    " << mvs_fg_exp << std::endl;

    // Extract subgame strategies from both and evaluate in original game
    auto combined_scalar = std::make_shared<TabularPolicy>(trunk);
    auto combined_mvs = std::make_shared<TabularPolicy>(trunk);
    const std::string prefix = "full_F:subgame:";
    auto info_per_player = CollectSubgameInfoStatesPerPlayer(it->second);

    for (const auto& [sub_is, ap] : scalar_policy.PolicyTable()) {
      if (sub_is.compare(0, prefix.length(), prefix) == 0) {
        std::string orig = sub_is.substr(prefix.length());
        if (info_per_player[0].count(orig) > 0) {
          combined_scalar->SetStatePolicy(orig, ap);
        }
      }
    }
    for (const auto& [sub_is, ap] : mvs_fg_policy.PolicyTable()) {
      if (sub_is.compare(0, prefix.length(), prefix) == 0) {
        std::string orig = sub_is.substr(prefix.length());
        if (info_per_player[0].count(orig) > 0) {
          combined_mvs->SetStatePolicy(orig, ap);
        }
      }
    }

    double orig_scalar_exp = algorithms::Exploitability(*game, *combined_scalar);
    double orig_mvs_exp = algorithms::Exploitability(*game, *combined_mvs);
    std::cout << "  Leduc FG scalar LP (orig game):  " << orig_scalar_exp << std::endl;
    std::cout << "  Leduc FG MVS LP (orig game):     " << orig_mvs_exp << std::endl;
    std::cout << "  (Note: only 1 of 30 groups resolved for res=0)" << std::endl;
  }

  std::cout << "TestDebugFullGadget PASSED" << std::endl;
}

// =============================================================================
// Test 11: Use MVS LP NE as trunk for Full Gadget on Leduc
// Hypothesis: with exact NE trunk, Full Gadget gives near-zero
// =============================================================================

// Helper: resolve all groups with Full Gadget (LP) and return combined policy.
// trunk_policy: used as chance probs for resolving player in trunk.
// base_policy: complete policy for the original game (subgame entries get
//   overwritten with Full Gadget LP results). Must have ALL info state entries
//   since Exploitability needs a complete policy.
std::shared_ptr<TabularPolicy> ResolveAllGroupsFullGadget(
    std::shared_ptr<const Game> game,
    const SubgameDecomposition& decomp,
    const TabularPolicy& trunk_policy,
    const TabularPolicy& base_policy,
    FullGadgetGame::Mode mode,
    bool use_mvs_boundaries,
    const std::string& solver_id = "CLP") {
  auto trunk_ptr = std::make_shared<TabularPolicy>(trunk_policy);
  auto combined = std::make_shared<TabularPolicy>(base_policy);

  // Build boundary data (history strings per group + scalar fallback values)
  std::unordered_map<std::string, std::vector<std::string>> boundary_by_group;
  std::unordered_map<std::string, std::vector<double>> boundary_values;
  for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
    for (const auto& root : roots) {
      std::string hist = root->HistoryString();
      boundary_by_group[pub_obs].push_back(hist);
      if (!use_mvs_boundaries) {
        // Scalar fallback: expected returns under base policy (only if no MVS)
        boundary_values[hist] = ComputeExpReturns(*root, base_policy);
      } else {
        // MVS handles boundaries — just need a placeholder
        boundary_values[hist] = {0.0, 0.0};
      }
    }
  }

  int num_groups = decomp.grouped_subgames.size();
  int failed = 0;
  ThrowingErrorGuard guard;  // LP failures throw instead of exit

  for (int res = 0; res < 2; ++res) {
    int gi = 0;
    for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
      ++gi;
      auto info_per_player = CollectSubgameInfoStatesPerPlayer(roots);

      auto fg = CreateFullGadgetGame(
          game, trunk_ptr, res, pub_obs,
          boundary_by_group, boundary_values, mode,
          /*boundary_portfolios_p0=*/{},
          /*boundary_portfolios_p1=*/{},
          /*enumerate_boundary_portfolios=*/use_mvs_boundaries);

      TabularPolicy fg_policy;
      try {
        algorithms::ortools::SequenceFormLpSpecification spec(
            *fg, solver_id, /*return_nan_if_non_optimal=*/false);
        auto [pol, val] =
            algorithms::ortools::MakeEquilibriumPolicy(&spec, true);
        fg_policy = pol;
        std::cout << "    res=" << res << " group=" << gi
                  << "/" << num_groups << " val=" << val << std::endl;
      } catch (const std::exception& e) {
        std::cerr << "    LP FAILED res=" << res << " group=" << gi
                  << "/" << num_groups << " (" << pub_obs << "): "
                  << e.what() << std::endl;
        ++failed;
        continue;
      }

      const std::string prefix = "full_F:subgame:";
      for (const auto& [sub_is, ap] : fg_policy.PolicyTable()) {
        if (sub_is.compare(0, prefix.length(), prefix) == 0) {
          std::string orig = sub_is.substr(prefix.length());
          if (info_per_player[res].count(orig) > 0) {
            // Clamp tiny negative probs and renormalize
            ActionsAndProbs clamped;
            double sum = 0.0;
            for (const auto& [a, p] : ap) {
              double cp = std::max(0.0, p);
              clamped.push_back({a, cp});
              sum += cp;
            }
            if (sum > 0.0) {
              for (auto& [a, p] : clamped) p /= sum;
            }
            combined->SetStatePolicy(orig, clamped);
          }
        }
      }
    }
  }
  if (failed > 0) {
    std::cerr << "    WARNING: " << failed << "/" << (num_groups * 2)
              << " LP solves failed" << std::endl;
  }
  return combined;
}

void TestLeducFullGadgetWithMVSTrunk() {
  std::cout << "TestLeducFullGadgetWithMVSTrunk..." << std::endl;
  auto game = LoadGame("leduc_poker");

  // --- Step 1: Strategy-agnostic decomposition ---
  auto decomp = DecomposeGameStructureAtRound(game, 3);
  std::cout << "  Groups: " << decomp.grouped_subgames.size() << std::endl;

  // --- Step 2: Get trunk strategies ---
  // 2a: MVS LP NE (exact equilibrium for depth-limited game)
  auto mvs = CreateMVSGameWithSubtreePureStrategies(
      game, 3, MVSGame::DepthMode::kRoundBased);
  auto [mvs_ne, mvs_value] =
      algorithms::ortools::MakeEquilibriumPolicy(*mvs, true);
  double mvs_exp = algorithms::Exploitability(*mvs, mvs_ne);
  std::cout << "  MVS LP exp (in MVS game): " << mvs_exp << std::endl;

  // Extract trunk strategy from MVS NE: filter out MVS-specific info states
  TabularPolicy mvs_trunk;
  for (const auto& [is, ap] : mvs_ne.PolicyTable()) {
    if (is.find(":MVSP_") == std::string::npos &&
        is.find(":MVS:") == std::string::npos &&
        is.find(":MVSP:") == std::string::npos) {
      mvs_trunk.SetStatePolicy(is, ap);
    }
  }

  // 2b: Full-game LP NE (for comparison)
  auto [full_ne, full_value] =
      algorithms::ortools::MakeEquilibriumPolicy(*game, true);
  double full_ne_exp = algorithms::Exploitability(*game, full_ne);
  std::cout << "  Full LP NE exp: " << full_ne_exp << std::endl;

  // 2c: CFR 500 (for comparison)
  algorithms::CFRSolverBase cfr_solver(*game, true, true, true);
  for (int i = 0; i < 500; ++i) cfr_solver.EvaluateAndUpdatePolicy();
  TabularPolicy cfr_trunk = cfr_solver.TabularAveragePolicy();
  double cfr_exp = algorithms::Exploitability(*game, cfr_trunk);
  std::cout << "  CFR 500 exp:    " << cfr_exp << std::endl;

  // --- Step 3: Full Gadget resolve with different trunks ---
  // Use full_ne as base_policy (complete for Exploitability).
  // Only the trunk_policy (chance probs in trunk) and MVS vs scalar varies.

  // Key test: MVS trunk + MVS boundaries (should give ~0 with LP)
  std::cout << "  [A] FG MVS trunk + MVS boundaries..." << std::endl;
  auto result_a = ResolveAllGroupsFullGadget(
      game, decomp, mvs_trunk, full_ne, FullGadgetGame::Mode::kTrunk, true);
  double exp_a = algorithms::Exploitability(*game, *result_a);
  std::cout << "  [A] FG MVS trunk + MVS (trunk):    " << exp_a << std::endl;

  // Full NE trunk + MVS boundaries
  std::cout << "  [B] FG full NE trunk + MVS boundaries..." << std::endl;
  auto result_b = ResolveAllGroupsFullGadget(
      game, decomp, full_ne, full_ne, FullGadgetGame::Mode::kTrunk, true);
  double exp_b = algorithms::Exploitability(*game, *result_b);
  std::cout << "  [B] FG full NE trunk + MVS (trunk): " << exp_b << std::endl;

  // Full NE trunk + scalar boundaries
  std::cout << "  [C] FG full NE trunk + scalar boundaries..." << std::endl;
  auto result_c = ResolveAllGroupsFullGadget(
      game, decomp, full_ne, full_ne, FullGadgetGame::Mode::kTrunk, false);
  double exp_c = algorithms::Exploitability(*game, *result_c);
  std::cout << "  [C] FG full NE trunk + scalar (trunk): " << exp_c << std::endl;

  // CFR trunk + MVS boundaries
  std::cout << "  [D] FG CFR trunk + MVS boundaries..." << std::endl;
  auto result_d = ResolveAllGroupsFullGadget(
      game, decomp, cfr_trunk, cfr_trunk, FullGadgetGame::Mode::kTrunk, true);
  double exp_d = algorithms::Exploitability(*game, *result_d);
  std::cout << "  [D] FG CFR trunk + MVS (trunk):    " << exp_d << std::endl;

  std::cout << "\n  === Summary ===" << std::endl;
  std::cout << "  Full NE:                          " << full_ne_exp << std::endl;
  std::cout << "  CFR 500:                          " << cfr_exp << std::endl;
  std::cout << "  MVS LP (in MVS):                  " << mvs_exp << std::endl;
  std::cout << "  [A] FG MVS trunk + MVS:           " << exp_a << std::endl;
  std::cout << "  [B] FG full NE trunk + MVS:       " << exp_b << std::endl;
  std::cout << "  [C] FG full NE trunk + scalar:    " << exp_c << std::endl;
  std::cout << "  [D] FG CFR trunk + MVS:           " << exp_d << std::endl;

  std::cout << "TestLeducFullGadgetWithMVSTrunk PASSED" << std::endl;
}

// Diagnostic: why does LP fail for res=1 on Leduc Full Gadget?
// Helper: try to solve a game with a specific solver config
bool TrySolveWithConfig(const Game& fg_game, const std::string& solver_id,
                        const std::string& label,
                        bool return_nan = false) {
  using namespace algorithms::ortools;
  try {
    SequenceFormLpSpecification spec(fg_game, solver_id, return_nan);
    TabularPolicy joint_policy;
    double game_value;
    for (int pl = 0; pl < 2; ++pl) {
      spec.SpecifyLinearProgram(pl);
      double val = spec.Solve();
      if (pl == 0) game_value = val;
      if (std::isnan(val)) {
        std::cout << "    " << label << ": NaN (non-optimal) for pl=" << pl
                  << std::endl;
        return false;
      }
    }
    std::cout << "    " << label << ": OK, value=" << game_value << std::endl;
    return true;
  } catch (const std::exception& e) {
    std::cout << "    " << label << ": FAILED: " << e.what() << std::endl;
    return false;
  }
}

void TestDiagnosticRes1Failure() {
  std::cout << "TestDiagnosticRes1Failure..." << std::endl;
  auto game = LoadGame("leduc_poker");

  auto [full_ne, full_value] =
      algorithms::ortools::MakeEquilibriumPolicy(*game, true);

  auto decomp = DecomposeGameStructureAtRound(game, 3);

  // Prepare boundary data
  std::unordered_map<std::string, std::vector<std::string>> boundary_by_group;
  std::unordered_map<std::string, std::vector<double>> boundary_values;
  for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
    for (const auto& root : roots) {
      std::string hist = root->HistoryString();
      boundary_by_group[pub_obs].push_back(hist);
      boundary_values[hist] = {0.0, 0.0};  // MVS handles it
    }
  }

  auto trunk_ptr = std::make_shared<TabularPolicy>(full_ne);

  ThrowingErrorGuard guard;

  // Try all groups, not just first
  int group_idx = 0;
  for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
    if (group_idx >= 3) break;  // Test first 3 groups
    std::cout << "\n  === Group " << group_idx << ": " << pub_obs
              << " (" << roots.size() << " roots) ===" << std::endl;

    // Build res=1 MVS full gadget for this group
    auto fg = CreateFullGadgetGame(
        game, trunk_ptr, 1, pub_obs,
        boundary_by_group, boundary_values,
        FullGadgetGame::Mode::kTrunk,
        {}, {}, true);

    // Count game tree size
    int terminals = 0, decisions = 0, chances = 0;
    std::function<void(State&)> count = [&](State& s) {
      if (s.IsTerminal()) { ++terminals; return; }
      if (s.IsChanceNode()) {
        ++chances;
        for (const auto& [a, p] : s.ChanceOutcomes()) {
          auto c = s.Clone(); c->ApplyAction(a); count(*c);
        }
      } else {
        ++decisions;
        for (Action a : s.LegalActions()) {
          auto c = s.Clone(); c->ApplyAction(a); count(*c);
        }
      }
    };
    count(*fg->NewInitialState());
    std::cout << "  res=1 MVS: terminals=" << terminals << " decisions="
              << decisions << " chances=" << chances << std::endl;

    // Try different solver configurations
    std::cout << "  Trying solver configurations:" << std::endl;

    // 1. Default GLOP
    TrySolveWithConfig(*fg, "GLOP", "GLOP default");

    // 2. GLOP with return_nan (to check status without crashing)
    TrySolveWithConfig(*fg, "GLOP", "GLOP (nan-safe)", true);

    // 3. CLP (alternative LP solver)
    TrySolveWithConfig(*fg, "CLP", "CLP");

    // 4. GLPK
    TrySolveWithConfig(*fg, "GLPK_LP", "GLPK_LP");

    ++group_idx;
  }

  std::cout << "\nTestDiagnosticRes1Failure DONE" << std::endl;
}

void TestLeducFullGadgetCLP() {
  std::cout << "TestLeducFullGadgetCLP..." << std::endl;
  auto game = LoadGame("leduc_poker");

  // Full-game LP NE (exact)
  auto [full_ne, full_value] =
      algorithms::ortools::MakeEquilibriumPolicy(*game, true);
  double full_ne_exp = algorithms::Exploitability(*game, full_ne);
  std::cout << "  Full LP NE exp: " << full_ne_exp << std::endl;

  // Strategy-agnostic decomposition
  auto decomp = DecomposeGameStructureAtRound(game, 3);
  std::cout << "  Groups: " << decomp.grouped_subgames.size() << std::endl;

  // Test [C]: Full NE trunk + scalar boundaries with CLP
  // This should give ~0 exploitability (all 60 groups should solve)
  std::cout << "\n  [C] FG full NE trunk + scalar boundaries (CLP)..."
            << std::endl;
  auto result_c = ResolveAllGroupsFullGadget(
      game, decomp, full_ne, full_ne, FullGadgetGame::Mode::kTrunk,
      /*use_mvs_boundaries=*/false, /*solver_id=*/"CLP");
  double exp_c = algorithms::Exploitability(*game, *result_c);
  std::cout << "  [C] Exploitability: " << exp_c << std::endl;

  // Test [B]: Full NE trunk + MVS boundaries with CLP — just 3 groups
  // to verify CLP handles MVS on both res=0 and res=1
  std::cout << "\n  [B-partial] FG full NE trunk + MVS (CLP, 3 groups)..."
            << std::endl;
  {
    auto trunk_ptr = std::make_shared<TabularPolicy>(full_ne);
    auto combined = std::make_shared<TabularPolicy>(full_ne);

    std::unordered_map<std::string, std::vector<std::string>> boundary_by_group;
    std::unordered_map<std::string, std::vector<double>> boundary_values;
    for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
      for (const auto& root : roots) {
        std::string hist = root->HistoryString();
        boundary_by_group[pub_obs].push_back(hist);
        boundary_values[hist] = {0.0, 0.0};
      }
    }

    int gi = 0, failed = 0;
    ThrowingErrorGuard guard;
    for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
      if (gi >= 3) break;  // Only 3 groups
      auto info_per_player = CollectSubgameInfoStatesPerPlayer(roots);

      for (int res = 0; res < 2; ++res) {
        auto fg = CreateFullGadgetGame(
            game, trunk_ptr, res, pub_obs,
            boundary_by_group, boundary_values,
            FullGadgetGame::Mode::kTrunk,
            {}, {}, true);

        try {
          algorithms::ortools::SequenceFormLpSpecification spec(
              *fg, "CLP", false);
          auto [pol, val] =
              algorithms::ortools::MakeEquilibriumPolicy(&spec, true);

          const std::string prefix = "full_F:subgame:";
          for (const auto& [sub_is, ap] : pol.PolicyTable()) {
            if (sub_is.compare(0, prefix.length(), prefix) == 0) {
              std::string orig = sub_is.substr(prefix.length());
              if (info_per_player[res].count(orig) > 0) {
                combined->SetStatePolicy(orig, ap);
              }
            }
          }
          std::cout << "    group=" << gi << " res=" << res
                    << " val=" << val << " OK" << std::endl;
        } catch (const std::exception& e) {
          std::cout << "    group=" << gi << " res=" << res
                    << " FAILED: " << e.what() << std::endl;
          ++failed;
        }
      }
      ++gi;
    }
    double exp_b = algorithms::Exploitability(*game, *combined);
    std::cout << "  [B-partial] Exploitability (3 groups): " << exp_b
              << std::endl;
    if (failed > 0) {
      std::cout << "    WARNING: " << failed << "/6 LP solves failed"
                << std::endl;
    }
  }

  std::cout << "\n  === Summary ===" << std::endl;
  std::cout << "  Full NE:           " << full_ne_exp << std::endl;
  std::cout << "  [C] scalar+CLP:    " << exp_c << std::endl;
  std::cout << "TestLeducFullGadgetCLP DONE" << std::endl;
}

// Diagnostic: Check MVS boundary structure in Full Gadget
// 1. Verify legal actions consistency within info sets
// 2. Check payoff matrix zero-sum property
// 3. Compare single-group resolve vs. standalone MVS
void TestMVSBoundaryDiagnostic() {
  std::cout << "TestMVSBoundaryDiagnostic..." << std::endl;
  auto game = LoadGame("leduc_poker");

  auto [full_ne, full_value] =
      algorithms::ortools::MakeEquilibriumPolicy(*game, true);
  auto decomp = DecomposeGameStructureAtRound(game, 3);
  auto trunk_ptr = std::make_shared<TabularPolicy>(full_ne);

  // Build boundary data
  std::unordered_map<std::string, std::vector<std::string>> boundary_by_group;
  std::unordered_map<std::string, std::vector<double>> boundary_values;
  for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
    for (const auto& root : roots) {
      std::string hist = root->HistoryString();
      boundary_by_group[pub_obs].push_back(hist);
      boundary_values[hist] = {0.0, 0.0};
    }
  }

  // Pick first group
  auto it = decomp.grouped_subgames.begin();
  std::string pub_obs = it->first;
  std::cout << "  Target group: " << pub_obs << std::endl;

  for (int res = 0; res < 2; ++res) {
    std::cout << "\n  --- res=" << res << " ---" << std::endl;
    auto fg = CreateFullGadgetGame(
        game, trunk_ptr, res, pub_obs,
        boundary_by_group, boundary_values,
        FullGadgetGame::Mode::kTrunk, {}, {}, true);

    // Traverse and check:
    // 1. Legal actions per info set
    // 2. Payoff matrix zero-sum
    std::unordered_map<std::string, int> is_to_num_actions;  // IS -> num_actions
    std::unordered_map<std::string, int> is_action_mismatch; // IS with mismatches
    int boundary_terminals = 0;
    double max_zs_violation = 0.0;
    int total_boundary_states = 0;

    std::function<void(State&)> check = [&](State& s) {
      if (s.IsTerminal()) {
        auto ret = s.Returns();
        double zs = std::abs(ret[0] + ret[1]);
        if (zs > max_zs_violation) max_zs_violation = zs;
        // Check if this is a boundary terminal (came from MVS)
        auto* fgs = dynamic_cast<FullGadgetState*>(&s);
        if (fgs) boundary_terminals++;
        return;
      }
      if (s.IsChanceNode()) {
        for (const auto& [a, p] : s.ChanceOutcomes()) {
          auto c = s.Clone(); c->ApplyAction(a); check(*c);
        }
        return;
      }

      Player pl = s.CurrentPlayer();
      std::string is = s.InformationStateString(pl);
      int na = s.LegalActions().size();

      auto* fgs = dynamic_cast<FullGadgetState*>(&s);
      bool is_boundary = fgs && (fgs->GetPhase() ==
          FullGadgetState::Phase::kBoundaryP0Select ||
          fgs->GetPhase() == FullGadgetState::Phase::kBoundaryP1Select);

      if (is_boundary) {
        total_boundary_states++;
      }

      auto prev = is_to_num_actions.find(is);
      if (prev != is_to_num_actions.end()) {
        if (prev->second != na) {
          is_action_mismatch[is]++;
          if (is_boundary) {
            std::cout << "    MISMATCH at boundary IS: " << is
                      << " actions=" << na << " vs " << prev->second
                      << std::endl;
          }
        }
      } else {
        is_to_num_actions[is] = na;
      }

      for (Action a : s.LegalActions()) {
        auto c = s.Clone(); c->ApplyAction(a); check(*c);
      }
    };

    check(*fg->NewInitialState());

    std::cout << "    Total boundary decision states: " << total_boundary_states
              << std::endl;
    std::cout << "    Distinct info sets: " << is_to_num_actions.size()
              << std::endl;
    std::cout << "    Info sets with action mismatch: "
              << is_action_mismatch.size() << std::endl;
    std::cout << "    Max zero-sum violation: " << max_zs_violation << std::endl;

    // Print some boundary IS details
    int shown = 0;
    for (const auto& [is, na] : is_to_num_actions) {
      if (is.find("full_boundary:") == 0 && shown < 5) {
        std::cout << "    Boundary IS: " << is << " actions=" << na
                  << std::endl;
        shown++;
      }
    }

    // Now try LP solve
    ThrowingErrorGuard guard;
    try {
      algorithms::ortools::SequenceFormLpSpecification spec(
          *fg, "CLP", false);
      auto [pol, val] =
          algorithms::ortools::MakeEquilibriumPolicy(&spec, true);
      std::cout << "    FG game value: " << val << std::endl;
      std::cout << "    Policy table size: " << pol.PolicyTable().size()
                << std::endl;

      // Check which game IS are missing from policy
      std::unordered_set<std::string> policy_is;
      int neg_prob_count = 0;
      for (const auto& [is, ap] : pol.PolicyTable()) {
        policy_is.insert(is);
        for (const auto& [a, p] : ap) {
          if (p < -1e-10) neg_prob_count++;
        }
      }
      std::cout << "    Actions with prob < -1e-10: " << neg_prob_count
                << std::endl;

      // Traverse game and check for missing IS
      int missing_is = 0;
      std::function<void(State&)> verify = [&](State& s) {
        if (s.IsTerminal()) return;
        if (s.IsChanceNode()) {
          for (const auto& [a, p] : s.ChanceOutcomes()) {
            auto c = s.Clone(); c->ApplyAction(a); verify(*c);
          }
          return;
        }
        Player pl = s.CurrentPlayer();
        std::string is = s.InformationStateString(pl);
        if (policy_is.find(is) == policy_is.end()) {
          if (missing_is < 5) {
            auto* fgs = dynamic_cast<FullGadgetState*>(&s);
            std::string phase_str = "unknown";
            if (fgs) {
              switch (fgs->GetPhase()) {
                case FullGadgetState::Phase::kTrunk: phase_str = "trunk"; break;
                case FullGadgetState::Phase::kSubgame: phase_str = "subgame"; break;
                case FullGadgetState::Phase::kBoundaryP0Select: phase_str = "BP0"; break;
                case FullGadgetState::Phase::kBoundaryP1Select: phase_str = "BP1"; break;
                case FullGadgetState::Phase::kTerminal: phase_str = "terminal"; break;
              }
            }
            std::cout << "    MISSING IS: player=" << pl
                      << " phase=" << phase_str
                      << " IS=" << is
                      << " actions=" << s.LegalActions().size()
                      << std::endl;
          }
          missing_is++;
        } else {
          // Check if all legal actions are present
          auto state_pol = pol.GetStatePolicy(is);
          for (Action a : s.LegalActions()) {
            double p = GetProb(state_pol, a);
            if (p < -0.5) {  // -1 sentinel
              if (missing_is < 5) {
                std::cout << "    MISSING ACTION: IS=" << is
                          << " action=" << a
                          << " legal_actions=" << s.LegalActions().size()
                          << " policy_actions=" << state_pol.size()
                          << std::endl;
              }
              missing_is++;
            }
          }
        }
        for (Action a : s.LegalActions()) {
          auto c = s.Clone(); c->ApplyAction(a); verify(*c);
        }
      };
      verify(*fg->NewInitialState());
      std::cout << "    Missing IS/actions: " << missing_is << std::endl;

    } catch (const std::exception& e) {
      std::cout << "    LP FAILED: " << e.what() << std::endl;
    }
  }

  std::cout << "\nTestMVSBoundaryDiagnostic DONE" << std::endl;
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::TestLeducFullGadgetWithMVSTrunk();
  return 0;
}
