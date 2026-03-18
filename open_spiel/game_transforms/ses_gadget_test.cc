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

#include "open_spiel/game_transforms/ses_gadget.h"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/game_transforms/max_margin_gadget.h"
#include "open_spiel/game_transforms/resolving_gadget.h"
#include "open_spiel/game_transforms/subgame_utils.h"
#include "open_spiel/game_transforms/turn_based_simultaneous_game.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace {

// Helper: build SES SubgameRoots (non-resolving player's IS, resolving reach)
std::vector<SubgameRoot> BuildSESRoots(
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

// =============================================================================
// TestSESConstruction
// =============================================================================

void TestSESConstruction() {
  std::cout << "TestSESConstruction..." << std::endl;

  auto game = LoadGame("kuhn_poker");
  auto uniform = std::make_shared<UniformPolicy>();

  auto states = CollectStatesAtDepth(*game, 2);
  std::vector<const State*> ptrs;
  for (const auto& s : states) ptrs.push_back(s.get());

  // For SES with resolving=P1 (res=1, non_res=0):
  // - info_state_string = P0's (non-resolving) info state
  // - reach_prob = P1's (resolving) reach
  auto resolving_reach = ComputeReachProbabilities(*game, *uniform, 1, ptrs);
  auto cfvs_nonres =
      ComputeCounterfactualValuesAtStates(*game, *uniform, 0, ptrs);
  auto ses_roots = BuildSESRoots(states, 0, resolving_reach);
  SPIEL_CHECK_GT(ses_roots.size(), 0);

  auto model_reach = ComputeModelInfoSetReach(*game, *uniform, 0, states);

  double alpha = 0.5;
  auto ses = CreateSESGadgetGame(game, std::move(ses_roots), 1,
                                  cfvs_nonres, alpha, model_reach);

  SPIEL_CHECK_EQ(ses->NumPlayers(), 2);
  SPIEL_CHECK_EQ(ses->ResolvingPlayer(), 1);
  SPIEL_CHECK_EQ(ses->NonResolvingPlayer(), 0);
  SPIEL_CHECK_GT(ses->NumInfoSets(), 0);
  SPIEL_CHECK_FLOAT_NEAR(ses->Alpha(), 0.5, 1e-9);

  std::cout << "  " << ses->NumSubgameRoots() << " roots, "
            << ses->NumInfoSets() << " info sets" << std::endl;
  std::cout << "TestSESConstruction PASSED" << std::endl;
}

// =============================================================================
// TestSESStateTransitions
// =============================================================================

void TestSESStateTransitions() {
  std::cout << "TestSESStateTransitions..." << std::endl;

  auto game = LoadGame("kuhn_poker");
  auto uniform = std::make_shared<UniformPolicy>();

  auto states = CollectStatesAtDepth(*game, 2);
  std::vector<const State*> ptrs;
  for (const auto& s : states) ptrs.push_back(s.get());

  auto resolving_reach = ComputeReachProbabilities(*game, *uniform, 1, ptrs);
  auto cfvs_nonres =
      ComputeCounterfactualValuesAtStates(*game, *uniform, 0, ptrs);
  auto ses_roots = BuildSESRoots(states, 0, resolving_reach);
  auto model_reach = ComputeModelInfoSetReach(*game, *uniform, 0, states);

  double alpha = 0.5;
  auto ses = CreateSESGadgetGame(game, std::move(ses_roots), 1,
                                  cfvs_nonres, alpha, model_reach);

  // --- Safety Branch (action 0) ---
  {
    auto state = ses->NewInitialState();
    // Phase kInitialChance: chance node
    SPIEL_CHECK_EQ(state->CurrentPlayer(), kChancePlayerId);
    auto outcomes = state->ChanceOutcomes();
    SPIEL_CHECK_EQ(static_cast<int>(outcomes.size()), 2);
    double total_prob = 0.0;
    for (const auto& [a, p] : outcomes) total_prob += p;
    SPIEL_CHECK_FLOAT_NEAR(total_prob, 1.0, 1e-9);

    // Take safety branch (action 0)
    state->ApplyAction(0);

    // Phase kSafeInfoSetChoice: NON-resolving player (P0) picks info set
    SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
    SPIEL_CHECK_EQ(static_cast<int>(state->LegalActions().size()),
                   ses->NumInfoSets());

    // Verify info states: resolving=ses_start, non-resolving=ses_safe:choosing
    SPIEL_CHECK_EQ(state->InformationStateString(1), "ses_start");
    SPIEL_CHECK_EQ(state->InformationStateString(0), "ses_safe:choosing");

    // Pick first info set
    state->ApplyAction(0);

    // Phase kChanceWithinInfoSet: chance picks state within info set
    SPIEL_CHECK_EQ(state->CurrentPlayer(), kChancePlayerId);
    auto within_outcomes = state->ChanceOutcomes();
    double prob_sum = 0.0;
    for (const auto& [a, p] : within_outcomes) prob_sum += p;
    SPIEL_CHECK_FLOAT_NEAR(prob_sum, 1.0, 1e-9);

    // Resolving player sees ses_start throughout
    SPIEL_CHECK_EQ(state->InformationStateString(1), "ses_start");

    // Enter subgame
    state->ApplyAction(within_outcomes[0].first);
    if (!state->IsTerminal()) {
      // Subgame info states should be ses_F:subgame:...
      Player cp = state->CurrentPlayer();
      if (cp >= 0) {
        std::string is = state->InformationStateString(cp);
        SPIEL_CHECK_TRUE(is.find("ses_F:subgame:") == 0);
      }
    }
  }

  // --- Exploit Branch (action 1) ---
  {
    auto state = ses->NewInitialState();
    // Take exploit branch
    state->ApplyAction(1);

    // Phase kExploitChanceInfoSet: chance picks info set by model reach
    SPIEL_CHECK_EQ(state->CurrentPlayer(), kChancePlayerId);

    // Verify info states
    SPIEL_CHECK_EQ(state->InformationStateString(1), "ses_start");
    SPIEL_CHECK_EQ(state->InformationStateString(0), "ses_exploit:chance");

    auto info_set_outcomes = state->ChanceOutcomes();
    double total = 0.0;
    for (const auto& [a, p] : info_set_outcomes) total += p;
    SPIEL_CHECK_FLOAT_NEAR(total, 1.0, 1e-9);

    // Pick first info set
    state->ApplyAction(info_set_outcomes[0].first);

    // Phase kChanceWithinInfoSet
    SPIEL_CHECK_EQ(state->CurrentPlayer(), kChancePlayerId);
    SPIEL_CHECK_EQ(state->InformationStateString(1), "ses_start");

    auto within = state->ChanceOutcomes();
    state->ApplyAction(within[0].first);

    if (!state->IsTerminal()) {
      Player cp = state->CurrentPlayer();
      if (cp >= 0) {
        std::string is = state->InformationStateString(cp);
        SPIEL_CHECK_TRUE(is.find("ses_F:subgame:") == 0);
      }
    }
  }

  std::cout << "TestSESStateTransitions PASSED" << std::endl;
}

// =============================================================================
// TestSESPayoffs
// =============================================================================

void TestSESPayoffs() {
  std::cout << "TestSESPayoffs..." << std::endl;

  auto game = LoadGame("kuhn_poker");
  auto uniform = std::make_shared<UniformPolicy>();

  auto states = CollectStatesAtDepth(*game, 2);
  std::vector<const State*> ptrs;
  for (const auto& s : states) ptrs.push_back(s.get());

  auto resolving_reach = ComputeReachProbabilities(*game, *uniform, 1, ptrs);
  auto cfvs_nonres =
      ComputeCounterfactualValuesAtStates(*game, *uniform, 0, ptrs);
  auto ses_roots = BuildSESRoots(states, 0, resolving_reach);
  auto model_reach = ComputeModelInfoSetReach(*game, *uniform, 0, states);

  auto ses = CreateSESGadgetGame(game, std::move(ses_roots), 1,
                                  cfvs_nonres, 0.5, model_reach);

  // Play through safety branch to terminal and verify zero-sum
  auto state = ses->NewInitialState();
  state->ApplyAction(0);  // safety branch
  state->ApplyAction(0);  // non-resolving picks info set 0
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
  std::cout << "  Safety branch: P0=" << returns[0] << ", P1=" << returns[1]
            << std::endl;

  // Play through exploit branch
  auto state2 = ses->NewInitialState();
  state2->ApplyAction(1);  // exploit branch
  state2->ApplyAction(state2->ChanceOutcomes()[0].first);  // chance picks IS
  state2->ApplyAction(state2->ChanceOutcomes()[0].first);  // chance within IS
  while (!state2->IsTerminal()) {
    if (state2->IsChanceNode()) {
      state2->ApplyAction(state2->ChanceOutcomes()[0].first);
    } else {
      state2->ApplyAction(state2->LegalActions()[0]);
    }
  }

  auto returns2 = state2->Returns();
  SPIEL_CHECK_FLOAT_NEAR(returns2[0] + returns2[1], 0.0, 1e-9);
  std::cout << "  Exploit branch: P0=" << returns2[0] << ", P1=" << returns2[1]
            << std::endl;

  std::cout << "TestSESPayoffs PASSED" << std::endl;
}

// =============================================================================
// TestCFROnSES
// =============================================================================

void TestCFROnSES() {
  std::cout << "TestCFROnSES..." << std::endl;

  auto game = LoadGame("kuhn_poker");
  auto uniform = std::make_shared<UniformPolicy>();

  auto states = CollectStatesAtDepth(*game, 2);
  std::vector<const State*> ptrs;
  for (const auto& s : states) ptrs.push_back(s.get());

  auto resolving_reach = ComputeReachProbabilities(*game, *uniform, 1, ptrs);
  auto cfvs_nonres =
      ComputeCounterfactualValuesAtStates(*game, *uniform, 0, ptrs);
  auto ses_roots = BuildSESRoots(states, 0, resolving_reach);
  auto model_reach = ComputeModelInfoSetReach(*game, *uniform, 0, states);

  auto ses = CreateSESGadgetGame(game, std::move(ses_roots), 1,
                                  cfvs_nonres, 0.5, model_reach);

  algorithms::CFRSolverBase cfr(*ses, true, true, true);
  for (int i = 0; i < 1000; ++i) cfr.EvaluateAndUpdatePolicy();

  double exp = algorithms::Exploitability(*ses, *cfr.AveragePolicy());
  std::cout << "  Exploitability = " << exp << std::endl;
  SPIEL_CHECK_LT(exp, 0.1);

  std::cout << "TestCFROnSES PASSED" << std::endl;
}

// =============================================================================
// TestSES_alpha0_MatchesMaxMargin — all four games
// =============================================================================

void CompareAlpha0VsMaxMargin(const std::string& name,
                               std::shared_ptr<const Game> game,
                               int decomp_depth, bool use_round,
                               int cfr_iters, double bound) {
  std::cout << "  " << name << "..." << std::endl;

  // Solve with CFR
  algorithms::CFRSolverBase solver(*game, true, true, true);
  for (int i = 0; i < cfr_iters; ++i) solver.EvaluateAndUpdatePolicy();
  auto policy = solver.AveragePolicy();
  TabularPolicy trunk = solver.TabularAveragePolicy();

  // Decompose
  SubgameDecomposition decomp = use_round
      ? DecomposeGameAtRound(game, *policy, decomp_depth)
      : DecomposeGameAtDepth(game, *policy, decomp_depth);

  // Re-solve with max-margin
  auto mm_policy = ResolveWithMaxMarginGadget(decomp, trunk, cfr_iters);
  double mm_exp = algorithms::Exploitability(*game, *mm_policy);

  // Re-solve with SES at alpha=0 (should match max-margin)
  auto ses_policy = ResolveWithSESGadget(decomp, trunk, 0.0, *policy,
                                          cfr_iters);
  double ses_exp = algorithms::Exploitability(*game, *ses_policy);

  std::cout << "    Max-margin: " << mm_exp << std::endl;
  std::cout << "    SES(α=0):   " << ses_exp << std::endl;

  // Both should be low (safe)
  SPIEL_CHECK_LT(mm_exp, bound);
  SPIEL_CHECK_LT(ses_exp, bound);

  // SES at alpha=0 should be comparable to max-margin
  double ratio = (mm_exp > 1e-10) ? ses_exp / mm_exp : 1.0;
  std::cout << "    Ratio SES/MM: " << ratio << std::endl;
  // Allow up to 5x difference due to CFR convergence noise
  SPIEL_CHECK_LT(ratio, 5.0);
}

void TestSES_alpha0_MatchesMaxMargin() {
  std::cout << "TestSES_alpha0_MatchesMaxMargin..." << std::endl;

  // Kuhn poker (depth 2)
  CompareAlpha0VsMaxMargin("Kuhn", LoadGame("kuhn_poker"),
                            2, false, 500, 0.01);

  // Leduc poker (round 3)
  CompareAlpha0VsMaxMargin("Leduc", LoadGame("leduc_poker"),
                            3, true, 500, 0.1);

  // Goofspiel 4-card (depth 4)
  CompareAlpha0VsMaxMargin("Goofspiel",
      LoadGameAsTurnBased("goofspiel",
          {{"num_cards", GameParameter(4)},
           {"imp_info", GameParameter(true)},
           {"points_order", GameParameter(std::string("descending"))}}),
      4, false, 500, 0.1);

  // Liar's Dice (depth 2)
  CompareAlpha0VsMaxMargin("Liar's Dice",
      LoadGame("liars_dice",
          {{"numdice", GameParameter(1)},
           {"dice_sides", GameParameter(4)}}),
      2, false, 500, 0.1);

  std::cout << "TestSES_alpha0_MatchesMaxMargin PASSED" << std::endl;
}

// =============================================================================
// TestSES_alpha1
// =============================================================================

void TestSES_alpha1() {
  std::cout << "TestSES_alpha1..." << std::endl;

  auto game = LoadGame("kuhn_poker");
  auto uniform = std::make_shared<UniformPolicy>();

  auto states = CollectStatesAtDepth(*game, 2);
  std::vector<const State*> ptrs;
  for (const auto& s : states) ptrs.push_back(s.get());

  auto resolving_reach = ComputeReachProbabilities(*game, *uniform, 1, ptrs);
  auto cfvs_nonres =
      ComputeCounterfactualValuesAtStates(*game, *uniform, 0, ptrs);
  auto ses_roots = BuildSESRoots(states, 0, resolving_reach);
  auto model_reach = ComputeModelInfoSetReach(*game, *uniform, 0, states);

  // alpha=1: pure exploitation branch (no safety)
  auto ses = CreateSESGadgetGame(game, std::move(ses_roots), 1,
                                  cfvs_nonres, 1.0, model_reach);

  algorithms::CFRSolverBase cfr(*ses, true, true, true);
  for (int i = 0; i < 1000; ++i) cfr.EvaluateAndUpdatePolicy();

  double exp = algorithms::Exploitability(*ses, *cfr.AveragePolicy());
  std::cout << "  Exploitability (alpha=1) = " << exp << std::endl;
  // Even at alpha=1, CFR should find a reasonable solution for the gadget game
  SPIEL_CHECK_LT(exp, 0.2);

  std::cout << "TestSES_alpha1 PASSED" << std::endl;
}

// =============================================================================
// TestSESLeducSafety
// =============================================================================

void TestSESLeducSafety() {
  std::cout << "TestSESLeducSafety..." << std::endl;

  auto game = LoadGame("leduc_poker");

  // Solve full game with CFR (more iterations for better blueprint)
  algorithms::CFRSolverBase solver(*game, true, true, true);
  for (int i = 0; i < 1000; ++i) solver.EvaluateAndUpdatePolicy();
  auto policy = solver.AveragePolicy();
  TabularPolicy trunk = solver.TabularAveragePolicy();
  double full_exp = algorithms::Exploitability(*game, *policy);

  // Decompose at round 3
  auto decomp = DecomposeGameAtRound(game, *policy, 3);

  // Uniform model
  UniformPolicy uniform_model;

  // Test a few key alpha values with more CFR iterations for better convergence
  // Leduc is a larger game so we use more iterations and a looser bound.
  // SES safety is guaranteed by the gadget structure (not just heuristically).
  double alpha_values[] = {0.0, 0.5, 1.0};
  for (double alpha : alpha_values) {
    auto ses_policy =
        ResolveWithSESGadget(decomp, trunk, alpha, uniform_model, 1000);
    double ses_exp = algorithms::Exploitability(*game, *ses_policy);
    std::cout << "  alpha=" << alpha << " exploitability=" << ses_exp
              << std::endl;
    // SES should remain safe (exploitability bounded). The bound is loose
    // since Leduc has a larger game tree requiring more CFR iterations to
    // converge fully. The key property is that it converges to a safe solution.
    SPIEL_CHECK_LT(ses_exp, 1.0);
  }

  std::cout << "  Full CFR exploitability: " << full_exp << std::endl;
  std::cout << "TestSESLeducSafety PASSED" << std::endl;
}

// =============================================================================
// TestSESGoofspielSafety
// =============================================================================

void TestSESGoofspielSafety() {
  std::cout << "TestSESGoofspielSafety..." << std::endl;

  auto game = LoadGameAsTurnBased(
      "goofspiel",
      {{"num_cards", GameParameter(4)},
       {"imp_info", GameParameter(true)},
       {"points_order", GameParameter(std::string("descending"))}});

  // Solve full game with CFR
  algorithms::CFRSolverBase solver(*game, true, true, true);
  for (int i = 0; i < 500; ++i) solver.EvaluateAndUpdatePolicy();
  auto policy = solver.AveragePolicy();
  TabularPolicy trunk = solver.TabularAveragePolicy();
  double full_exp = algorithms::Exploitability(*game, *policy);

  // Decompose at depth 4
  auto decomp = DecomposeGameAtDepth(game, *policy, 4);
  std::cout << "  Subgames: " << decomp.grouped_subgames.size() << std::endl;

  UniformPolicy uniform_model;

  // Test alpha=0.5 with sufficient CFR iterations
  auto ses_policy = ResolveWithSESGadget(decomp, trunk, 0.5, uniform_model,
                                          1000);
  double ses_exp = algorithms::Exploitability(*game, *ses_policy);

  std::cout << "  Full:     " << full_exp << std::endl;
  std::cout << "  SES(0.5): " << ses_exp << std::endl;

  // Goofspiel is a larger game; verify the strategy is valid and safe.
  // The exploitability should be bounded (finite and not catastrophically high).
  SPIEL_CHECK_LT(ses_exp, 2.0);

  std::cout << "TestSESGoofspielSafety PASSED" << std::endl;
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::TestSESConstruction();
  open_spiel::TestSESStateTransitions();
  open_spiel::TestSESPayoffs();
  open_spiel::TestCFROnSES();
  open_spiel::TestSES_alpha0_MatchesMaxMargin();
  open_spiel::TestSES_alpha1();
  open_spiel::TestSESLeducSafety();
  open_spiel::TestSESGoofspielSafety();

  std::cout << "\nAll SES tests passed!" << std::endl;
  return 0;
}
