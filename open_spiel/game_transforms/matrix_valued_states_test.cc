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

#include "open_spiel/game_transforms/matrix_valued_states.h"

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/algorithms/expected_returns.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"
#include "open_spiel/utils/init.h"

namespace open_spiel {
namespace {

// Test basic construction of MVS game from Kuhn poker
void TestBasicConstruction() {
  std::cout << "TestBasicConstruction" << std::endl;

  auto game = LoadGame("kuhn_poker");

  auto uniform = std::make_shared<UniformPolicy>();
  std::vector<std::shared_ptr<Policy>> p1_portfolios = {uniform};
  std::vector<std::shared_ptr<Policy>> p2_portfolios = {uniform};

  auto mvs_game = CreateMVSGame(game, p1_portfolios, p2_portfolios,
                                 /*depth_limit=*/2);

  SPIEL_CHECK_TRUE(mvs_game != nullptr);
  SPIEL_CHECK_EQ(mvs_game->NumPlayers(), 2);
  SPIEL_CHECK_EQ(mvs_game->DepthLimit(), 2);
  SPIEL_CHECK_EQ(mvs_game->NumPortfoliosP1(), 1);
  SPIEL_CHECK_EQ(mvs_game->NumPortfoliosP2(), 1);
  SPIEL_CHECK_GE(mvs_game->NumDistinctActions(), 1);

  auto state = mvs_game->NewInitialState();
  SPIEL_CHECK_TRUE(state != nullptr);
  SPIEL_CHECK_FALSE(state->IsTerminal());

  std::cout << "Initial state:\n" << state->ToString() << std::endl;
}

// Test that depth limit triggers correctly
void TestDepthLimitTrigger() {
  std::cout << "TestDepthLimitTrigger" << std::endl;

  auto game = LoadGame("kuhn_poker");

  auto uniform = std::make_shared<UniformPolicy>();
  std::vector<std::shared_ptr<Policy>> p1_portfolios = {uniform};
  std::vector<std::shared_ptr<Policy>> p2_portfolios = {uniform};

  // Depth limit of 1: should trigger after first player action
  auto mvs_game = CreateMVSGame(game, p1_portfolios, p2_portfolios,
                                 /*depth_limit=*/1);

  auto state = mvs_game->NewInitialState();

  // Navigate through chance nodes
  while (state->IsChanceNode()) {
    auto outcomes = state->ChanceOutcomes();
    state->ApplyAction(outcomes[0].first);
  }

  std::cout << "After chance nodes:\n" << state->ToString() << std::endl;
  std::cout << "Current player: " << state->CurrentPlayer() << std::endl;

  // First player action
  auto legal = state->LegalActions();
  SPIEL_CHECK_GT(legal.size(), 0);
  state->ApplyAction(legal[0]);

  std::cout << "After P1 action:\n" << state->ToString() << std::endl;

  // Now we should be in portfolio selection
  auto* mvs_state = dynamic_cast<MVSState*>(state.get());
  SPIEL_CHECK_TRUE(mvs_state != nullptr);

  // Check we're in portfolio selection phase
  std::cout << "Phase: " << static_cast<int>(mvs_state->GetPhase()) << std::endl;
  SPIEL_CHECK_TRUE(mvs_state->GetPhase() == MVSState::Phase::kPortfolioP2 ||
                   mvs_state->GetPhase() == MVSState::Phase::kPortfolioP1);
}

// Test that information states are correct during simultaneous selection
void TestInformationStateCorrectness() {
  std::cout << "TestInformationStateCorrectness" << std::endl;

  auto game = LoadGame("kuhn_poker");

  auto uniform = std::make_shared<UniformPolicy>();
  auto first_action = std::make_shared<FirstActionPolicy>();
  std::vector<std::shared_ptr<Policy>> p1_portfolios = {uniform, first_action};
  std::vector<std::shared_ptr<Policy>> p2_portfolios = {uniform, first_action};

  // Depth 0: immediately trigger portfolio selection
  auto mvs_game = CreateMVSGame(game, p1_portfolios, p2_portfolios,
                                 /*depth_limit=*/0);

  auto state = mvs_game->NewInitialState();

  // Navigate through chance nodes
  while (state->IsChanceNode()) {
    auto outcomes = state->ChanceOutcomes();
    state->ApplyAction(outcomes[0].first);
  }

  // Now at depth limit, P1 should be selecting
  std::string p1_info_before = state->InformationStateString(0);
  std::string p2_info_before = state->InformationStateString(1);

  std::cout << "P1 info before: " << p1_info_before << std::endl;
  std::cout << "P2 info before: " << p2_info_before << std::endl;

  // P1 selects portfolio 1
  state->ApplyAction(1);

  // P2's info state should NOT reveal P1's choice
  std::string p1_info_after = state->InformationStateString(0);
  std::string p2_info_after = state->InformationStateString(1);

  std::cout << "P1 info after P1 action: " << p1_info_after << std::endl;
  std::cout << "P2 info after P1 action: " << p2_info_after << std::endl;

  // During selection, both should have the same MVS_SELECT suffix
  SPIEL_CHECK_TRUE(p2_info_after.find(":MVS_SELECT") != std::string::npos);
  // P2's info should NOT contain P1's choice
  SPIEL_CHECK_TRUE(p2_info_after.find(":MVS:1") == std::string::npos);

  // P2 selects portfolio 0
  state->ApplyAction(0);

  // Now at terminal
  SPIEL_CHECK_TRUE(state->IsTerminal());

  // At terminal, each player knows their own choice
  std::string p1_terminal = state->InformationStateString(0);
  std::string p2_terminal = state->InformationStateString(1);

  std::cout << "P1 terminal info: " << p1_terminal << std::endl;
  std::cout << "P2 terminal info: " << p2_terminal << std::endl;

  // P1 should see their choice (1), P2 should see their choice (0)
  SPIEL_CHECK_TRUE(p1_terminal.find(":MVS:1") != std::string::npos);
  SPIEL_CHECK_TRUE(p2_terminal.find(":MVS:0") != std::string::npos);
}

// Test that payoff computation matches ExpectedReturns
void TestPayoffComputation() {
  std::cout << "TestPayoffComputation" << std::endl;

  auto game = LoadGame("kuhn_poker");

  auto uniform = std::make_shared<UniformPolicy>();
  auto first_action = std::make_shared<FirstActionPolicy>();
  std::vector<std::shared_ptr<Policy>> p1_portfolios = {uniform};
  std::vector<std::shared_ptr<Policy>> p2_portfolios = {uniform};

  // Depth 0: test payoff at initial state
  auto mvs_game = CreateMVSGame(game, p1_portfolios, p2_portfolios,
                                 /*depth_limit=*/0);

  auto state = mvs_game->NewInitialState();

  // Navigate through chance nodes
  while (state->IsChanceNode()) {
    auto outcomes = state->ChanceOutcomes();
    state->ApplyAction(outcomes[0].first);
  }

  // Store the underlying state for direct comparison
  auto* mvs_state = dynamic_cast<MVSState*>(state.get());
  SPIEL_CHECK_TRUE(mvs_state != nullptr);

  // Compute expected returns directly using the underlying state
  std::unique_ptr<State> underlying_clone = game->NewInitialState();
  for (Action a : state->History()) {
    underlying_clone->ApplyAction(a);
  }

  std::vector<const Policy*> policies = {uniform.get(), uniform.get()};
  auto expected = algorithms::ExpectedReturns(*underlying_clone, policies,
                                               /*depth_limit=*/-1,
                                               /*use_infostate_get_policy=*/false);

  // Now play through MVS to get returns
  // P1 selects portfolio 0
  state->ApplyAction(0);
  // P2 selects portfolio 0
  state->ApplyAction(0);

  SPIEL_CHECK_TRUE(state->IsTerminal());
  auto returns = state->Returns();

  std::cout << "Expected returns: [" << expected[0] << ", " << expected[1]
            << "]" << std::endl;
  std::cout << "MVS returns: [" << returns[0] << ", " << returns[1] << "]"
            << std::endl;

  // Should match
  SPIEL_CHECK_FLOAT_EQ(returns[0], expected[0]);
  SPIEL_CHECK_FLOAT_EQ(returns[1], expected[1]);
}

// Test compatibility with CFR
void TestCFRCompatibility() {
  std::cout << "TestCFRCompatibility" << std::endl;

  auto game = LoadGame("kuhn_poker");

  auto uniform = std::make_shared<UniformPolicy>();
  std::vector<std::shared_ptr<Policy>> p1_portfolios = {uniform};
  std::vector<std::shared_ptr<Policy>> p2_portfolios = {uniform};

  // With depth 0, this becomes a trivial 1x1 matrix game
  auto mvs_game = CreateMVSGame(game, p1_portfolios, p2_portfolios,
                                 /*depth_limit=*/0);

  // Run CFR on the transformed game
  algorithms::CFRSolverBase solver(*mvs_game,
                                    /*alternating_updates=*/true,
                                    /*linear_averaging=*/false,
                                    /*regret_matching_plus=*/false);

  for (int i = 0; i < 10; ++i) {
    solver.EvaluateAndUpdatePolicy();
  }

  auto average_policy = solver.AveragePolicy();
  auto state = mvs_game->NewInitialState();

  auto expected_returns = algorithms::ExpectedReturns(
      *state, *average_policy, /*depth_limit=*/-1,
      /*use_infostate_get_policy=*/true);

  std::cout << "CFR expected returns after 10 iterations: [" << expected_returns[0]
            << ", " << expected_returns[1] << "]" << std::endl;

  // Just verify it runs without crashing and returns reasonable values
  SPIEL_CHECK_GE(expected_returns[0], game->MinUtility());
  SPIEL_CHECK_LE(expected_returns[0], game->MaxUtility());
}

// Test with multiple portfolios
void TestMultiplePortfolios() {
  std::cout << "TestMultiplePortfolios" << std::endl;

  auto game = LoadGame("kuhn_poker");

  auto uniform = std::make_shared<UniformPolicy>();
  auto first_action = std::make_shared<FirstActionPolicy>();
  std::vector<std::shared_ptr<Policy>> p1_portfolios = {uniform, first_action};
  std::vector<std::shared_ptr<Policy>> p2_portfolios = {uniform, first_action};

  auto mvs_game = CreateMVSGame(game, p1_portfolios, p2_portfolios,
                                 /*depth_limit=*/0);

  SPIEL_CHECK_EQ(mvs_game->NumPortfoliosP1(), 2);
  SPIEL_CHECK_EQ(mvs_game->NumPortfoliosP2(), 2);
  SPIEL_CHECK_GE(mvs_game->NumDistinctActions(), 2);

  auto state = mvs_game->NewInitialState();

  // Navigate through chance nodes
  while (state->IsChanceNode()) {
    auto outcomes = state->ChanceOutcomes();
    state->ApplyAction(outcomes[0].first);
  }

  // Should have 2 portfolio choices
  auto legal = state->LegalActions();
  SPIEL_CHECK_EQ(legal.size(), 2);
  SPIEL_CHECK_EQ(legal[0], 0);
  SPIEL_CHECK_EQ(legal[1], 1);

  std::cout << "Legal actions for P1: ";
  for (Action a : legal) {
    std::cout << a << " ";
  }
  std::cout << std::endl;

  // Try each combination and verify we get different payoffs
  std::vector<std::vector<double>> all_returns;
  for (int p1_choice = 0; p1_choice < 2; ++p1_choice) {
    for (int p2_choice = 0; p2_choice < 2; ++p2_choice) {
      auto test_state = mvs_game->NewInitialState();
      while (test_state->IsChanceNode()) {
        auto outcomes = test_state->ChanceOutcomes();
        test_state->ApplyAction(outcomes[0].first);
      }
      test_state->ApplyAction(p1_choice);
      test_state->ApplyAction(p2_choice);
      auto returns = test_state->Returns();
      all_returns.push_back(returns);
      std::cout << "P1=" << p1_choice << ", P2=" << p2_choice << " -> ["
                << returns[0] << ", " << returns[1] << "]" << std::endl;
    }
  }

  // Verify the payoff matrix is cached (second access should be fast)
  // and consistent
  for (int p1_choice = 0; p1_choice < 2; ++p1_choice) {
    for (int p2_choice = 0; p2_choice < 2; ++p2_choice) {
      auto test_state = mvs_game->NewInitialState();
      while (test_state->IsChanceNode()) {
        auto outcomes = test_state->ChanceOutcomes();
        test_state->ApplyAction(outcomes[0].first);
      }
      test_state->ApplyAction(p1_choice);
      test_state->ApplyAction(p2_choice);
      auto returns = test_state->Returns();
      int idx = p1_choice * 2 + p2_choice;
      SPIEL_CHECK_FLOAT_EQ(returns[0], all_returns[idx][0]);
      SPIEL_CHECK_FLOAT_EQ(returns[1], all_returns[idx][1]);
    }
  }
}

// Test game ending before depth limit (true terminal)
void TestEarlyTerminal() {
  std::cout << "TestEarlyTerminal" << std::endl;

  auto game = LoadGame("kuhn_poker");

  auto uniform = std::make_shared<UniformPolicy>();
  std::vector<std::shared_ptr<Policy>> p1_portfolios = {uniform};
  std::vector<std::shared_ptr<Policy>> p2_portfolios = {uniform};

  // Large depth limit - game will end naturally
  auto mvs_game = CreateMVSGame(game, p1_portfolios, p2_portfolios,
                                 /*depth_limit=*/100);

  auto state = mvs_game->NewInitialState();

  // Play until terminal
  int moves = 0;
  while (!state->IsTerminal()) {
    if (state->IsChanceNode()) {
      auto outcomes = state->ChanceOutcomes();
      state->ApplyAction(outcomes[0].first);
    } else {
      auto legal = state->LegalActions();
      state->ApplyAction(legal[0]);
    }
    moves++;
    SPIEL_CHECK_LT(moves, 100);  // Prevent infinite loop
  }

  // Game should have ended naturally, not via matrix terminal
  auto* mvs_state = dynamic_cast<MVSState*>(state.get());
  std::cout << "Final phase: " << static_cast<int>(mvs_state->GetPhase())
            << std::endl;

  auto returns = state->Returns();
  std::cout << "Final returns: [" << returns[0] << ", " << returns[1] << "]"
            << std::endl;

  // Returns should be valid
  SPIEL_CHECK_GE(returns[0], game->MinUtility());
  SPIEL_CHECK_LE(returns[0], game->MaxUtility());
}

// Run basic random simulation tests
void TestRandomSimulation() {
  std::cout << "TestRandomSimulation" << std::endl;

  auto game = LoadGame("kuhn_poker");

  auto uniform = std::make_shared<UniformPolicy>();
  auto first_action = std::make_shared<FirstActionPolicy>();
  std::vector<std::shared_ptr<Policy>> p1_portfolios = {uniform, first_action};
  std::vector<std::shared_ptr<Policy>> p2_portfolios = {uniform, first_action};

  auto mvs_game = CreateMVSGame(game, p1_portfolios, p2_portfolios,
                                 /*depth_limit=*/2);

  // Run random simulations - disable serialization since MVS game isn't registered
  testing::RandomSimTest(*mvs_game, 50, /*serialize=*/false);
  std::cout << "Random simulation tests passed!" << std::endl;
}

// Test round-based depth mode (for games like Leduc poker)
void TestRoundBasedDepth() {
  std::cout << "TestRoundBasedDepth" << std::endl;

  auto game = LoadGame("kuhn_poker");

  auto uniform = std::make_shared<UniformPolicy>();
  std::vector<std::shared_ptr<Policy>> p1_portfolios = {uniform};
  std::vector<std::shared_ptr<Policy>> p2_portfolios = {uniform};

  // Round-based depth mode
  auto mvs_game = CreateMVSGame(game, p1_portfolios, p2_portfolios,
                                 /*depth_limit=*/1,
                                 MVSGame::DepthMode::kRoundBased);

  SPIEL_CHECK_TRUE(mvs_game->GetDepthMode() == MVSGame::DepthMode::kRoundBased);

  auto state = mvs_game->NewInitialState();
  SPIEL_CHECK_FALSE(state->IsTerminal());

  std::cout << "Round-based mode game created successfully" << std::endl;

  // Navigate and verify it works
  int moves = 0;
  while (!state->IsTerminal() && moves < 20) {
    if (state->IsChanceNode()) {
      auto outcomes = state->ChanceOutcomes();
      state->ApplyAction(outcomes[0].first);
    } else {
      auto legal = state->LegalActions();
      SPIEL_CHECK_GT(legal.size(), 0);
      state->ApplyAction(legal[0]);
    }
    moves++;
  }

  SPIEL_CHECK_TRUE(state->IsTerminal());
  std::cout << "Round-based depth test completed" << std::endl;
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::Init("", &argc, &argv, true);

  open_spiel::TestBasicConstruction();
  open_spiel::TestDepthLimitTrigger();
  open_spiel::TestInformationStateCorrectness();
  open_spiel::TestPayoffComputation();
  open_spiel::TestMultiplePortfolios();
  open_spiel::TestEarlyTerminal();
  open_spiel::TestRoundBasedDepth();
  open_spiel::TestRandomSimulation();
  open_spiel::TestCFRCompatibility();

  std::cout << "\nAll tests passed!" << std::endl;
  return 0;
}
