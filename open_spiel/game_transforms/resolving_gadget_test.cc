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

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/game_transforms/matrix_valued_states.h"
#include "open_spiel/game_transforms/subgame_utils.h"
#include "open_spiel/game_transforms/turn_based_simultaneous_game.h"
#include "open_spiel/game_transforms/unsafe_subgame.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace {

// =============================================================================
// Basic Tests
// =============================================================================

void TestGadgetConstruction() {
  std::cout << "TestGadgetConstruction..." << std::endl;

  auto game = LoadGame("kuhn_poker");
  auto uniform = std::make_shared<UniformPolicy>();

  auto states = CollectStatesAtDepth(*game, 2);
  std::vector<const State*> ptrs;
  for (const auto& s : states) ptrs.push_back(s.get());

  auto reaches = ComputeReachProbabilities(*game, *uniform, 0, ptrs);
  auto cfvs = ComputeCounterfactualValuesAtStates(*game, *uniform, 1, ptrs);
  auto roots = BuildSubgameRoots(states, 1, reaches);
  SPIEL_CHECK_GT(roots.size(), 0);

  auto gadget = CreateGadgetGame(game, std::move(roots), 1, cfvs);

  SPIEL_CHECK_EQ(gadget->NumPlayers(), 2);
  SPIEL_CHECK_EQ(gadget->ResolvingPlayer(), 1);
  SPIEL_CHECK_GT(gadget->NormalizationConstant(), 0);

  std::cout << "  " << gadget->NumSubgameRoots() << " roots, k="
            << gadget->NormalizationConstant() << std::endl;
  std::cout << "TestGadgetConstruction PASSED" << std::endl;
}

void TestGadgetStateTransitions() {
  std::cout << "TestGadgetStateTransitions..." << std::endl;

  auto game = LoadGame("kuhn_poker");
  auto uniform = std::make_shared<UniformPolicy>();

  auto states = CollectStatesAtDepth(*game, 2);
  std::vector<const State*> ptrs;
  for (const auto& s : states) ptrs.push_back(s.get());

  auto reaches = ComputeReachProbabilities(*game, *uniform, 0, ptrs);
  auto cfvs = ComputeCounterfactualValuesAtStates(*game, *uniform, 1, ptrs);
  auto gadget = CreateGadgetGame(
      game, BuildSubgameRoots(states, 1, reaches), 1, cfvs);

  // Phase 1: Chance
  auto state = gadget->NewInitialState();
  SPIEL_CHECK_TRUE(state->IsChanceNode());

  auto outcomes = state->ChanceOutcomes();
  double prob_sum = 0;
  for (const auto& [a, p] : outcomes) prob_sum += p;
  SPIEL_CHECK_FLOAT_NEAR(prob_sum, 1.0, 1e-9);

  // Phase 2: Gadget choice (T/F)
  state->ApplyAction(outcomes[0].first);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  auto actions = state->LegalActions();
  SPIEL_CHECK_EQ(actions.size(), 2);
  SPIEL_CHECK_EQ(actions[0], GadgetGame::kTerminateAction);
  SPIEL_CHECK_EQ(actions[1], GadgetGame::kFollowAction);

  // T -> terminal
  auto state_t = state->Clone();
  state_t->ApplyAction(GadgetGame::kTerminateAction);
  SPIEL_CHECK_TRUE(state_t->IsTerminal());

  // F -> subgame
  auto state_f = state->Clone();
  state_f->ApplyAction(GadgetGame::kFollowAction);
  if (!state_f->IsTerminal()) {
    SPIEL_CHECK_GE(state_f->CurrentPlayer(), 0);
  }

  std::cout << "TestGadgetStateTransitions PASSED" << std::endl;
}

void TestGadgetPayoffs() {
  std::cout << "TestGadgetPayoffs..." << std::endl;

  auto game = LoadGame("kuhn_poker");
  auto uniform = std::make_shared<UniformPolicy>();

  auto states = CollectStatesAtDepth(*game, 2);
  std::vector<const State*> ptrs;
  for (const auto& s : states) ptrs.push_back(s.get());

  auto reaches = ComputeReachProbabilities(*game, *uniform, 0, ptrs);
  auto cfvs = ComputeCounterfactualValuesAtStates(*game, *uniform, 1, ptrs);
  auto gadget = CreateGadgetGame(
      game, BuildSubgameRoots(states, 1, reaches), 1, cfvs);

  auto state = gadget->NewInitialState();
  auto outcomes = state->ChanceOutcomes();
  state->ApplyAction(outcomes[0].first);

  auto* gs = dynamic_cast<GadgetState*>(state.get());
  std::string info_state = gadget->GetRootInfoState(gs->SelectedRootIndex());

  // Take T action
  state->ApplyAction(GadgetGame::kTerminateAction);
  SPIEL_CHECK_TRUE(state->IsTerminal());

  auto returns = state->Returns();
  double expected_t = gadget->GetTerminatePayoff(info_state);
  SPIEL_CHECK_FLOAT_NEAR(returns[1], expected_t, 1e-9);
  SPIEL_CHECK_FLOAT_NEAR(returns[0], -expected_t, 1e-9);

  std::cout << "TestGadgetPayoffs PASSED" << std::endl;
}

// =============================================================================
// CFR on Gadget
// =============================================================================

void TestCFROnGadget() {
  std::cout << "TestCFROnGadget..." << std::endl;

  auto game = LoadGame("kuhn_poker");
  auto uniform = std::make_shared<UniformPolicy>();

  auto states = CollectStatesAtDepth(*game, 2);
  std::vector<const State*> ptrs;
  for (const auto& s : states) ptrs.push_back(s.get());

  auto reaches = ComputeReachProbabilities(*game, *uniform, 0, ptrs);
  auto cfvs = ComputeCounterfactualValuesAtStates(*game, *uniform, 1, ptrs);
  auto gadget = CreateGadgetGame(
      game, BuildSubgameRoots(states, 1, reaches), 1, cfvs);

  algorithms::CFRSolverBase solver(*gadget, true, true, true);
  for (int i = 0; i < 1000; ++i) solver.EvaluateAndUpdatePolicy();

  double exp = algorithms::Exploitability(*gadget, *solver.AveragePolicy());
  std::cout << "  Exploitability = " << exp << std::endl;
  SPIEL_CHECK_LT(exp, 0.1);

  std::cout << "TestCFROnGadget PASSED" << std::endl;
}

// =============================================================================
// MVS + Gadget: Leduc
// =============================================================================

void TestLeducMVSWithGadgetResolving() {
  std::cout << "TestLeducMVSWithGadgetResolving..." << std::endl;

  auto game = LoadGame("leduc_poker");

  // Solve full game for baseline
  algorithms::CFRSolverBase full_solver(*game, true, true, true);
  for (int i = 0; i < 500; ++i) full_solver.EvaluateAndUpdatePolicy();
  auto full_policy = full_solver.AveragePolicy();
  double full_exp = algorithms::Exploitability(*game, *full_policy);

  // Solve MVS trunk
  auto mvs_game = CreateMVSGameWithSubtreePureStrategies(
      game, 3, MVSGame::DepthMode::kRoundBased);
  algorithms::CFRSolverBase mvs_solver(*mvs_game, true, true, true);
  for (int i = 0; i < 500; ++i) mvs_solver.EvaluateAndUpdatePolicy();
  auto mvs_policy = mvs_solver.AveragePolicy();

  // Extract MVS trunk into original game's info state space
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

  // Build decomposition from MVS data
  SubgameDecomposition decomp;
  decomp.game = game;
  decomp.trunk_info_states = CollectInfoStateStringsBeforeRound(*game, 3);
  decomp.grouped_subgames = GroupStatesByPublicObservation(
      *game, CollectStatesAtRound(*game, 3));
  decomp.reach_probs[0] = ExtractReachProbsFromMVS(*mvs_game, *mvs_policy, 0);
  decomp.reach_probs[1] = ExtractReachProbsFromMVS(*mvs_game, *mvs_policy, 1);
  decomp.cfvs[0] = ExtractCFVsFromMVSSolution(*mvs_game, *mvs_policy, 0);
  decomp.cfvs[1] = ExtractCFVsFromMVSSolution(*mvs_game, *mvs_policy, 1);

  auto resolved = ResolveWithGadget(decomp, trunk, 500);
  double resolved_exp = algorithms::Exploitability(*game, *resolved);

  std::cout << "  Full:     " << full_exp << std::endl;
  std::cout << "  Combined: " << resolved_exp << std::endl;
  std::cout << "  Info states: " << resolved->PolicyTable().size()
            << std::endl;
  std::cout << "  Subgames: " << decomp.grouped_subgames.size() << std::endl;

  SPIEL_CHECK_LT(resolved_exp, 0.5);

  std::cout << "TestLeducMVSWithGadgetResolving PASSED" << std::endl;
}

// =============================================================================
// Safe vs Unsafe Re-Solving
// =============================================================================

void TestSafeVsUnsafeResolving() {
  std::cout << "TestSafeVsUnsafeResolving..." << std::endl;

  auto game = LoadGame("leduc_poker");

  // Solve full game
  algorithms::CFRSolverBase solver(*game, true, true, true);
  for (int i = 0; i < 500; ++i) solver.EvaluateAndUpdatePolicy();
  auto policy = solver.AveragePolicy();
  TabularPolicy trunk = solver.TabularAveragePolicy();
  double full_exp = algorithms::Exploitability(*game, *policy);

  // Decompose at round 3
  auto decomp = DecomposeGameAtRound(game, *policy, 3);

  // Re-solve with both methods
  auto safe = ResolveWithGadget(decomp, trunk, 500);
  auto unsafe = ResolveWithUnsafeSubgame(decomp, trunk, 500);

  double safe_exp = algorithms::Exploitability(*game, *safe);
  double unsafe_exp = algorithms::Exploitability(*game, *unsafe);

  std::cout << "  Full:   " << full_exp << std::endl;
  std::cout << "  Safe:   " << safe_exp << std::endl;
  std::cout << "  Unsafe: " << unsafe_exp << std::endl;

  SPIEL_CHECK_LT(safe_exp, 0.1);
  SPIEL_CHECK_LT(unsafe_exp, 0.5);

  std::cout << "TestSafeVsUnsafeResolving PASSED" << std::endl;
}

// =============================================================================
// MVS + Gadget: Goofspiel
// =============================================================================

void TestGoofspielMVSWithGadgetResolving() {
  std::cout << "TestGoofspielMVSWithGadgetResolving..." << std::endl;

  auto game = LoadGameAsTurnBased(
      "goofspiel",
      {{"num_cards", GameParameter(4)},
       {"imp_info", GameParameter(true)},
       {"points_order", GameParameter(std::string("descending"))}});

  // Solve full game for baseline
  algorithms::CFRSolverBase full_solver(*game, true, true, true);
  for (int i = 0; i < 500; ++i) full_solver.EvaluateAndUpdatePolicy();
  auto full_policy = full_solver.AveragePolicy();
  double full_exp = algorithms::Exploitability(*game, *full_policy);

  // Solve MVS trunk
  auto mvs_game = CreateMVSGameWithSubtreePureStrategies(
      game, 4, MVSGame::DepthMode::kActionBased);
  algorithms::CFRSolverBase mvs_solver(*mvs_game, true, true, true);
  for (int i = 0; i < 500; ++i) mvs_solver.EvaluateAndUpdatePolicy();
  auto mvs_policy = mvs_solver.AveragePolicy();

  // Extract MVS trunk into original game's info state space
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

  // Build decomposition from MVS data
  SubgameDecomposition decomp;
  decomp.game = game;
  decomp.trunk_info_states = CollectInfoStateStringsBeforeDepth(*game, 4);
  decomp.grouped_subgames = GroupStatesByPublicObservation(
      *game, CollectStatesAtDepth(*game, 4));
  decomp.reach_probs[0] = ExtractReachProbsFromMVS(*mvs_game, *mvs_policy, 0);
  decomp.reach_probs[1] = ExtractReachProbsFromMVS(*mvs_game, *mvs_policy, 1);
  decomp.cfvs[0] = ExtractCFVsFromMVSSolution(*mvs_game, *mvs_policy, 0);
  decomp.cfvs[1] = ExtractCFVsFromMVSSolution(*mvs_game, *mvs_policy, 1);

  auto resolved = ResolveWithGadget(decomp, trunk, 500);
  double resolved_exp = algorithms::Exploitability(*game, *resolved);

  std::cout << "  Full:     " << full_exp << std::endl;
  std::cout << "  Combined: " << resolved_exp << std::endl;
  std::cout << "  Subgames: " << decomp.grouped_subgames.size() << std::endl;

  SPIEL_CHECK_LT(resolved_exp, 0.05);

  std::cout << "TestGoofspielMVSWithGadgetResolving PASSED" << std::endl;
}

// =============================================================================
// Main
// =============================================================================

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::TestGadgetConstruction();
  open_spiel::TestGadgetStateTransitions();
  open_spiel::TestGadgetPayoffs();
  open_spiel::TestCFROnGadget();
  open_spiel::TestLeducMVSWithGadgetResolving();
  open_spiel::TestSafeVsUnsafeResolving();
  open_spiel::TestGoofspielMVSWithGadgetResolving();

  std::cout << "\nAll tests passed!" << std::endl;
  return 0;
}
