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

#include "open_spiel/game_transforms/unsafe_subgame.h"

#include <iostream>
#include <memory>
#include <vector>

#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/game_transforms/subgame_utils.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace {

void TestUnsafeSubgameConstruction() {
  std::cout << "TestUnsafeSubgameConstruction..." << std::endl;

  auto game = LoadGame("kuhn_poker");

  // Collect states at depth 2 (after first betting round)
  auto roots = CollectStatesAtDepth(*game, 2);
  SPIEL_CHECK_GT(roots.size(), 0);

  // Create uniform reach probabilities
  std::vector<double> reach_probs(roots.size(), 1.0 / roots.size());

  auto unsafe_game = CreateUnsafeSubgame(game, std::move(roots), reach_probs);

  SPIEL_CHECK_EQ(unsafe_game->NumPlayers(), 2);
  SPIEL_CHECK_GT(unsafe_game->NumSubgameRoots(), 0);
  SPIEL_CHECK_FLOAT_NEAR(unsafe_game->NormalizationConstant(), 1.0, 1e-9);

  std::cout << "  Created unsafe subgame with " << unsafe_game->NumSubgameRoots()
            << " roots" << std::endl;
  std::cout << "TestUnsafeSubgameConstruction PASSED" << std::endl;
}

void TestUnsafeSubgameTransitions() {
  std::cout << "TestUnsafeSubgameTransitions..." << std::endl;

  auto game = LoadGame("kuhn_poker");
  auto roots = CollectStatesAtDepth(*game, 2);
  std::vector<double> reach_probs(roots.size(), 1.0);

  auto unsafe_game = CreateUnsafeSubgame(game, std::move(roots), reach_probs);

  // Test initial state
  auto state = unsafe_game->NewInitialState();
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  SPIEL_CHECK_EQ(state->CurrentPlayer(), kChancePlayerId);

  // Check chance outcomes sum to 1
  auto outcomes = state->ChanceOutcomes();
  SPIEL_CHECK_GT(outcomes.size(), 0);
  double prob_sum = 0;
  for (const auto& [action, prob] : outcomes) {
    prob_sum += prob;
  }
  SPIEL_CHECK_FLOAT_NEAR(prob_sum, 1.0, 1e-9);

  // Apply chance action
  state->ApplyAction(outcomes[0].first);

  // Should now be in subgame (not chance anymore, unless subgame has chance)
  auto* unsafe_state = dynamic_cast<UnsafeSubgameState*>(state.get());
  SPIEL_CHECK_EQ(static_cast<int>(unsafe_state->GetPhase()),
                 static_cast<int>(UnsafeSubgameState::Phase::kSubgame));

  std::cout << "TestUnsafeSubgameTransitions PASSED" << std::endl;
}

void TestCFROnUnsafeSubgame() {
  std::cout << "TestCFROnUnsafeSubgame..." << std::endl;

  auto game = LoadGame("kuhn_poker");
  auto roots = CollectStatesAtDepth(*game, 2);

  // Use uniform reaches
  std::vector<double> reach_probs(roots.size(), 1.0);

  auto unsafe_game = CreateUnsafeSubgame(game, std::move(roots), reach_probs);

  // Run CFR on the unsafe subgame
  algorithms::CFRSolverBase solver(
      *unsafe_game,
      /*alternating_updates=*/true,
      /*linear_averaging=*/true,
      /*regret_matching_plus=*/true);

  for (int i = 0; i < 1000; ++i) {
    solver.EvaluateAndUpdatePolicy();
  }

  auto avg_policy = solver.AveragePolicy();
  double exploitability = algorithms::Exploitability(*unsafe_game, *avg_policy);

  std::cout << "  CFR on unsafe subgame: exploitability = " << exploitability
            << std::endl;

  // The unsafe subgame itself should be solvable
  SPIEL_CHECK_LT(exploitability, 0.01);

  std::cout << "TestCFROnUnsafeSubgame PASSED" << std::endl;
}

void TestInfoStateStrings() {
  std::cout << "TestInfoStateStrings..." << std::endl;

  auto game = LoadGame("kuhn_poker");
  auto roots = CollectStatesAtDepth(*game, 2);
  std::vector<double> reach_probs(roots.size(), 1.0);

  auto unsafe_game = CreateUnsafeSubgame(game, std::move(roots), reach_probs);

  auto state = unsafe_game->NewInitialState();

  // At chance node
  std::string info_p0 = state->InformationStateString(0);
  std::string info_p1 = state->InformationStateString(1);
  SPIEL_CHECK_EQ(info_p0, "unsafe_start");
  SPIEL_CHECK_EQ(info_p1, "unsafe_start");

  // Apply chance and check info states have prefix
  state->ApplyAction(0);
  info_p0 = state->InformationStateString(0);
  info_p1 = state->InformationStateString(1);
  SPIEL_CHECK_TRUE(info_p0.find("unsafe:subgame:") == 0);
  SPIEL_CHECK_TRUE(info_p1.find("unsafe:subgame:") == 0);

  std::cout << "TestInfoStateStrings PASSED" << std::endl;
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::TestUnsafeSubgameConstruction();
  open_spiel::TestUnsafeSubgameTransitions();
  open_spiel::TestCFROnUnsafeSubgame();
  open_spiel::TestInfoStateStrings();

  std::cout << "\nAll unsafe_subgame tests passed!" << std::endl;
  return 0;
}
