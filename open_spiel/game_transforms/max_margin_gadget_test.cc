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

#include "open_spiel/game_transforms/max_margin_gadget.h"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/algorithms/expected_returns.h"
#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/game_transforms/resolving_gadget.h"
#include "open_spiel/game_transforms/subgame_utils.h"
#include "open_spiel/game_transforms/turn_based_simultaneous_game.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace {

// =============================================================================
// Basic Tests
// =============================================================================

void TestMaxMarginConstruction() {
  std::cout << "TestMaxMarginConstruction..." << std::endl;

  auto game = LoadGame("kuhn_poker");
  auto uniform = std::make_shared<UniformPolicy>();

  auto states = CollectStatesAtDepth(*game, 2);
  std::vector<const State*> ptrs;
  for (const auto& s : states) ptrs.push_back(s.get());

  auto reaches = ComputeReachProbabilities(*game, *uniform, 0, ptrs);
  auto cfvs = ComputeCounterfactualValuesAtStates(*game, *uniform, 1, ptrs);
  auto roots = BuildSubgameRoots(states, 1, reaches);
  SPIEL_CHECK_GT(roots.size(), 0);

  auto mm = CreateMaxMarginGadgetGame(game, std::move(roots), 1, cfvs);

  SPIEL_CHECK_EQ(mm->NumPlayers(), 2);
  SPIEL_CHECK_EQ(mm->ResolvingPlayer(), 1);
  SPIEL_CHECK_GT(mm->NumInfoSets(), 0);

  std::cout << "  " << mm->NumSubgameRoots() << " roots, "
            << mm->NumInfoSets() << " info sets" << std::endl;
  std::cout << "TestMaxMarginConstruction PASSED" << std::endl;
}

void TestMaxMarginStateTransitions() {
  std::cout << "TestMaxMarginStateTransitions..." << std::endl;

  auto game = LoadGame("kuhn_poker");
  auto uniform = std::make_shared<UniformPolicy>();

  auto states = CollectStatesAtDepth(*game, 2);
  std::vector<const State*> ptrs;
  for (const auto& s : states) ptrs.push_back(s.get());

  auto reaches = ComputeReachProbabilities(*game, *uniform, 0, ptrs);
  auto cfvs = ComputeCounterfactualValuesAtStates(*game, *uniform, 1, ptrs);
  auto mm = CreateMaxMarginGadgetGame(
      game, BuildSubgameRoots(states, 1, reaches), 1, cfvs);

  // Phase 1: InfoSetChoice — resolving player picks info set
  auto state = mm->NewInitialState();
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_EQ(static_cast<int>(state->LegalActions().size()),
                 mm->NumInfoSets());

  // Phase 2: Chance — picks state within info set
  state->ApplyAction(0);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), kChancePlayerId);
  auto outcomes = state->ChanceOutcomes();
  double prob_sum = 0;
  for (const auto& [a, p] : outcomes) prob_sum += p;
  SPIEL_CHECK_FLOAT_NEAR(prob_sum, 1.0, 1e-9);

  // Phase 3: Subgame
  state->ApplyAction(outcomes[0].first);
  if (!state->IsTerminal()) {
    SPIEL_CHECK_GE(state->CurrentPlayer(), 0);
  }

  std::cout << "TestMaxMarginStateTransitions PASSED" << std::endl;
}

void TestMaxMarginPayoffs() {
  std::cout << "TestMaxMarginPayoffs..." << std::endl;

  auto game = LoadGame("kuhn_poker");
  auto uniform = std::make_shared<UniformPolicy>();

  auto states = CollectStatesAtDepth(*game, 2);
  std::vector<const State*> ptrs;
  for (const auto& s : states) ptrs.push_back(s.get());

  auto reaches = ComputeReachProbabilities(*game, *uniform, 0, ptrs);
  auto cfvs = ComputeCounterfactualValuesAtStates(*game, *uniform, 1, ptrs);
  auto mm = CreateMaxMarginGadgetGame(
      game, BuildSubgameRoots(states, 1, reaches), 1, cfvs);

  // Play through to terminal and verify zero-sum
  auto state = mm->NewInitialState();
  state->ApplyAction(0);  // info set choice
  state->ApplyAction(state->ChanceOutcomes()[0].first);  // chance
  while (!state->IsTerminal()) {
    if (state->IsChanceNode()) {
      state->ApplyAction(state->ChanceOutcomes()[0].first);
    } else {
      state->ApplyAction(state->LegalActions()[0]);
    }
  }

  auto returns = state->Returns();
  SPIEL_CHECK_FLOAT_NEAR(returns[0] + returns[1], 0.0, 1e-9);

  std::cout << "  Returns: P0=" << returns[0] << ", P1=" << returns[1]
            << std::endl;
  std::cout << "TestMaxMarginPayoffs PASSED" << std::endl;
}

// =============================================================================
// Blueprint CFV Test
// =============================================================================

void TestMaxMarginBlueprintCFVsAreZero() {
  std::cout << "TestMaxMarginBlueprintCFVsAreZero..." << std::endl;

  // Key test: under the blueprint that produced the CFVs, the expected
  // shifted value at each info set must be exactly 0.
  // Proof: E[v|I] = (1/W) * sum_h reach(h)*v(h) - CFV/W = 0

  auto game = LoadGame("kuhn_poker");
  algorithms::CFRSolverBase solver(*game, true, true, true);
  for (int i = 0; i < 500; ++i) solver.EvaluateAndUpdatePolicy();
  auto policy = solver.AveragePolicy();

  auto states = CollectStatesAtDepth(*game, 2);
  std::vector<const State*> ptrs;
  for (const auto& s : states) ptrs.push_back(s.get());

  auto reaches = ComputeReachProbabilities(*game, *policy, 0, ptrs);
  auto cfvs = ComputeCounterfactualValuesAtStates(
      *game, *policy, 1, ptrs, reaches);

  auto mm = CreateMaxMarginGadgetGame(
      game, BuildSubgameRoots(states, 1, reaches), 1, cfvs);

  // Per-info-set check: expected shifted value must be ~0
  for (int i = 0; i < mm->NumInfoSets(); ++i) {
    const auto& is = mm->InfoSetForAction(i);
    double shift = mm->GetValueShift(is);
    double W = mm->GetInfoSetReachSum(is);

    double ev = 0.0;
    for (int idx : mm->RootsForInfoSet(i)) {
      double prob = mm->GetRootReachProbability(idx) / W;
      auto root = mm->CloneRootState(idx);
      auto er = algorithms::ExpectedReturns(*root, *policy, -1);
      ev += prob * (er[1] - shift);
    }

    std::cout << "    " << is << ": CFV=" << cfvs[is] << ", W=" << W
              << ", shift=" << shift << ", E[v]=" << ev << std::endl;
    SPIEL_CHECK_FLOAT_NEAR(ev, 0.0, 1e-6);
  }

  // Overall game value under blueprint should also be ~0
  auto mm_pol = std::make_shared<TabularPolicy>();
  {
    ActionsAndProbs ap;
    double u = 1.0 / mm->NumInfoSets();
    for (int i = 0; i < mm->NumInfoSets(); ++i) ap.push_back({i, u});
    mm_pol->SetStatePolicy("mm_start", ap);
  }
  // Fill subgame info states with blueprint policy
  for (int i = 0; i < mm->NumSubgameRoots(); ++i) {
    auto root = mm->CloneRootState(i);
    std::function<void(const State&)> fill = [&](const State& s) {
      if (s.IsTerminal()) return;
      if (s.IsChanceNode()) {
        for (const auto& [a, p] : s.ChanceOutcomes()) {
          auto c = s.Clone(); c->ApplyAction(a); fill(*c);
        }
      } else {
        Player pl = s.CurrentPlayer();
        auto ap = policy->GetStatePolicy(s, pl);
        if (!ap.empty()) {
          mm_pol->SetStatePolicy(
              absl::StrCat("mm_F:subgame:", s.InformationStateString(pl)), ap);
        }
        for (Action a : s.LegalActions()) {
          auto c = s.Clone(); c->ApplyAction(a); fill(*c);
        }
      }
    };
    fill(*root);
  }

  auto er = algorithms::ExpectedReturns(*mm->NewInitialState(), *mm_pol, -1);
  std::cout << "  Overall: P0=" << er[0] << ", P1=" << er[1] << std::endl;
  SPIEL_CHECK_FLOAT_NEAR(er[1], 0.0, 1e-6);
  SPIEL_CHECK_FLOAT_NEAR(er[0] + er[1], 0.0, 1e-9);

  std::cout << "TestMaxMarginBlueprintCFVsAreZero PASSED" << std::endl;
}

// =============================================================================
// CFR on Max-Margin Gadget
// =============================================================================

void TestCFROnMaxMargin() {
  std::cout << "TestCFROnMaxMargin..." << std::endl;

  auto game = LoadGame("kuhn_poker");
  auto uniform = std::make_shared<UniformPolicy>();

  auto states = CollectStatesAtDepth(*game, 2);
  std::vector<const State*> ptrs;
  for (const auto& s : states) ptrs.push_back(s.get());

  auto reaches = ComputeReachProbabilities(*game, *uniform, 0, ptrs);
  auto cfvs = ComputeCounterfactualValuesAtStates(*game, *uniform, 1, ptrs);
  auto mm = CreateMaxMarginGadgetGame(
      game, BuildSubgameRoots(states, 1, reaches), 1, cfvs);

  algorithms::CFRSolverBase cfr(*mm, true, true, true);
  for (int i = 0; i < 1000; ++i) cfr.EvaluateAndUpdatePolicy();

  double exp = algorithms::Exploitability(*mm, *cfr.AveragePolicy());
  std::cout << "  Exploitability = " << exp << std::endl;
  SPIEL_CHECK_LT(exp, 0.1);

  std::cout << "TestCFROnMaxMargin PASSED" << std::endl;
}

// =============================================================================
// Leduc: Safe Gadget vs Max-Margin Gadget
// =============================================================================

void TestLeducSafeVsMaxMargin() {
  std::cout << "TestLeducSafeVsMaxMargin..." << std::endl;

  auto game = LoadGame("leduc_poker");

  // Solve full game
  algorithms::CFRSolverBase solver(*game, true, true, true);
  for (int i = 0; i < 500; ++i) solver.EvaluateAndUpdatePolicy();
  auto policy = solver.AveragePolicy();
  TabularPolicy trunk = solver.TabularAveragePolicy();
  double full_exp = algorithms::Exploitability(*game, *policy);

  // Decompose at round 3
  auto decomp = DecomposeGameAtRound(game, *policy, 3);

  // Re-solve with both gadgets
  auto safe = ResolveWithGadget(decomp, trunk, 500);
  auto mm = ResolveWithMaxMarginGadget(decomp, trunk, 500);

  double safe_exp = algorithms::Exploitability(*game, *safe);
  double mm_exp = algorithms::Exploitability(*game, *mm);

  std::cout << "  Full:       " << full_exp << std::endl;
  std::cout << "  Safe:       " << safe_exp << std::endl;
  std::cout << "  Max-margin: " << mm_exp << std::endl;

  SPIEL_CHECK_LT(safe_exp, 0.1);
  SPIEL_CHECK_LT(mm_exp, 0.1);

  std::cout << "TestLeducSafeVsMaxMargin PASSED" << std::endl;
}

// =============================================================================
// Goofspiel: Safe Gadget vs Max-Margin Gadget
// =============================================================================

void TestGoofspielSafeVsMaxMargin() {
  std::cout << "TestGoofspielSafeVsMaxMargin..." << std::endl;

  auto game = LoadGameAsTurnBased(
      "goofspiel",
      {{"num_cards", GameParameter(4)},
       {"imp_info", GameParameter(true)},
       {"points_order", GameParameter(std::string("descending"))}});

  // Solve full game
  algorithms::CFRSolverBase solver(*game, true, true, true);
  for (int i = 0; i < 500; ++i) solver.EvaluateAndUpdatePolicy();
  auto policy = solver.AveragePolicy();
  TabularPolicy trunk = solver.TabularAveragePolicy();
  double full_exp = algorithms::Exploitability(*game, *policy);

  // Decompose at depth 4 (two full rounds of bidding)
  auto decomp = DecomposeGameAtDepth(game, *policy, 4);

  std::cout << "  Subgames: " << decomp.grouped_subgames.size() << std::endl;

  // Re-solve with both gadgets
  auto safe = ResolveWithGadget(decomp, trunk, 500);
  auto mm = ResolveWithMaxMarginGadget(decomp, trunk, 500);

  double safe_exp = algorithms::Exploitability(*game, *safe);
  double mm_exp = algorithms::Exploitability(*game, *mm);

  std::cout << "  Full:       " << full_exp << std::endl;
  std::cout << "  Safe:       " << safe_exp << std::endl;
  std::cout << "  Max-margin: " << mm_exp << std::endl;

  SPIEL_CHECK_LT(safe_exp, 0.1);
  SPIEL_CHECK_LT(mm_exp, 0.1);

  std::cout << "TestGoofspielSafeVsMaxMargin PASSED" << std::endl;
}

// =============================================================================
// Main
// =============================================================================

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::TestMaxMarginConstruction();
  open_spiel::TestMaxMarginStateTransitions();
  open_spiel::TestMaxMarginPayoffs();
  open_spiel::TestMaxMarginBlueprintCFVsAreZero();
  open_spiel::TestCFROnMaxMargin();
  open_spiel::TestLeducSafeVsMaxMargin();
  open_spiel::TestGoofspielSafeVsMaxMargin();

  std::cout << "\nAll tests passed!" << std::endl;
  return 0;
}
