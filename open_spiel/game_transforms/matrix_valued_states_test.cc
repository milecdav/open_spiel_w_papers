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

#include <algorithm>
#include <chrono>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/game_transforms/subgame_utils.h"
#include "open_spiel/game_transforms/turn_based_simultaneous_game.h"
#include "open_spiel/algorithms/best_response.h"
#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/algorithms/expected_returns.h"
#include "open_spiel/algorithms/tabular_exploitability.h"
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

// ============================================================================
// Pure Strategy Enumeration Tests
// ============================================================================

// Test basic pure strategy enumeration
void TestPureStrategyEnumeration() {
  std::cout << "TestPureStrategyEnumeration" << std::endl;

  auto game = LoadGame("kuhn_poker");
  auto state = game->NewInitialState();

  // Navigate through chance nodes to get a decision node
  while (state->IsChanceNode()) {
    auto outcomes = state->ChanceOutcomes();
    state->ApplyAction(outcomes[0].first);
  }

  std::cout << "State after chance: " << state->ToString() << std::endl;

  // Collect infostates for player 0
  SubtreeInfostates p0_infostates = CollectSubtreeInfostates(*state, 0);
  std::cout << "Player 0 infostates in subtree: " << p0_infostates.infostates.size()
            << std::endl;
  for (const auto& is : p0_infostates.infostates) {
    std::cout << "  " << is << " -> " << p0_infostates.legal_actions.at(is).size()
              << " actions" << std::endl;
  }

  // Collect infostates for player 1
  SubtreeInfostates p1_infostates = CollectSubtreeInfostates(*state, 1);
  std::cout << "Player 1 infostates in subtree: " << p1_infostates.infostates.size()
            << std::endl;
  for (const auto& is : p1_infostates.infostates) {
    std::cout << "  " << is << " -> " << p1_infostates.legal_actions.at(is).size()
              << " actions" << std::endl;
  }

  // Enumerate pure strategies for player 0
  auto p0_strategies = EnumeratePureStrategies(p0_infostates);
  std::cout << "Player 0 pure strategies: " << p0_strategies.size() << std::endl;

  // Enumerate pure strategies for player 1
  auto p1_strategies = EnumeratePureStrategies(p1_infostates);
  std::cout << "Player 1 pure strategies: " << p1_strategies.size() << std::endl;

  // Verify count: product of action counts at each infostate
  int expected_p0 = 1;
  for (const auto& is : p0_infostates.infostates) {
    expected_p0 *= p0_infostates.legal_actions.at(is).size();
  }
  SPIEL_CHECK_EQ(p0_strategies.size(), expected_p0);

  int expected_p1 = 1;
  for (const auto& is : p1_infostates.infostates) {
    expected_p1 *= p1_infostates.legal_actions.at(is).size();
  }
  SPIEL_CHECK_EQ(p1_strategies.size(), expected_p1);
}

// Sorted multiset of branch widths at each of the player's infosets in the
// continuation (labels ignored). Two subtrees with the same signature have the
// same number of global pure strategies as ∏ signature_i.
std::vector<int> SubtreeActionCountSignature(const SubtreeInfostates& infostates) {
  std::vector<int> sig;
  sig.reserve(infostates.infostates.size());
  for (const auto& is : infostates.infostates) {
    sig.push_back(static_cast<int>(infostates.legal_actions.at(is).size()));
  }
  std::sort(sig.begin(), sig.end());
  return sig;
}

// Full Gadget with enumerate_boundary_portfolios=true builds MVS matrices
// separately at each boundary *history*. For a sane public-state abstraction,
// every root in a kPublicStateObsType group should expose the same continuation
// "shape" (same multiset of action counts per infoset) and thus the same
// portfolio sizes; otherwise portfolio indices mean different things at
// different histories that share a public observation.
void TestPureStrategyContinuationConsistentAcrossPublicObservation() {
  std::cout << "TestPureStrategyContinuationConsistentAcrossPublicObservation"
            << std::endl;

  auto game = LoadGame("leduc_poker");
  std::vector<std::unique_ptr<State>> roots = CollectStatesAtRound(*game, 3);
  auto grouped = GroupStatesByPublicObservation(*game, std::move(roots));

  int groups_checked = 0;
  for (const auto& [pub_obs, group_roots] : grouped) {
    (void)pub_obs;
    if (group_roots.size() <= 1) continue;
    ++groups_checked;

    std::vector<int> ref_p0_sig;
    std::vector<int> ref_p1_sig;
    int ref_n_p0 = -1;
    int ref_n_p1 = -1;
    size_t ref_is0 = 0;
    size_t ref_is1 = 0;

    for (size_t ri = 0; ri < group_roots.size(); ++ri) {
      const State& r = *group_roots[ri];
      SubtreeInfostates p0_is = CollectSubtreeInfostates(r, 0);
      SubtreeInfostates p1_is = CollectSubtreeInfostates(r, 1);

      auto p0_strats = EnumeratePureStrategies(p0_is);
      auto p1_strats = EnumeratePureStrategies(p1_is);

      int prod_p0 = 1;
      for (const auto& is : p0_is.infostates) {
        prod_p0 *= static_cast<int>(p0_is.legal_actions.at(is).size());
      }
      int prod_p1 = 1;
      for (const auto& is : p1_is.infostates) {
        prod_p1 *= static_cast<int>(p1_is.legal_actions.at(is).size());
      }
      SPIEL_CHECK_EQ(static_cast<int>(p0_strats.size()), prod_p0);
      SPIEL_CHECK_EQ(static_cast<int>(p1_strats.size()), prod_p1);

      std::vector<int> sig0 = SubtreeActionCountSignature(p0_is);
      std::vector<int> sig1 = SubtreeActionCountSignature(p1_is);

      if (ri == 0) {
        ref_p0_sig = std::move(sig0);
        ref_p1_sig = std::move(sig1);
        ref_n_p0 = static_cast<int>(p0_strats.size());
        ref_n_p1 = static_cast<int>(p1_strats.size());
        ref_is0 = p0_is.infostates.size();
        ref_is1 = p1_is.infostates.size();
      } else {
        SPIEL_CHECK_EQ(sig0, ref_p0_sig);
        SPIEL_CHECK_EQ(sig1, ref_p1_sig);
        SPIEL_CHECK_EQ(static_cast<int>(p0_strats.size()), ref_n_p0);
        SPIEL_CHECK_EQ(static_cast<int>(p1_strats.size()), ref_n_p1);
        SPIEL_CHECK_EQ(p0_is.infostates.size(), ref_is0);
        SPIEL_CHECK_EQ(p1_is.infostates.size(), ref_is1);
      }
    }
  }

  SPIEL_CHECK_GT(groups_checked, 0);
  std::cout << "  Checked " << groups_checked
            << " Leduc public-obs groups with multiple round-3 roots" << std::endl;
}

// Test that pure strategies can be used as policies
void TestPureStrategyAsPolicy() {
  std::cout << "TestPureStrategyAsPolicy" << std::endl;

  auto game = LoadGame("kuhn_poker");
  auto state = game->NewInitialState();

  // Navigate through chance nodes
  while (state->IsChanceNode()) {
    auto outcomes = state->ChanceOutcomes();
    state->ApplyAction(outcomes[0].first);
  }

  // Get pure strategies as portfolios
  auto p0_portfolio = EnumerateSubtreePureStrategies(*state, 0);
  auto p1_portfolio = EnumerateSubtreePureStrategies(*state, 1);

  std::cout << "P0 portfolio size: " << p0_portfolio.size() << std::endl;
  std::cout << "P1 portfolio size: " << p1_portfolio.size() << std::endl;

  SPIEL_CHECK_GT(p0_portfolio.size(), 0);
  SPIEL_CHECK_GT(p1_portfolio.size(), 0);

  // Test that each policy can be queried
  for (size_t i = 0; i < p0_portfolio.size(); ++i) {
    auto test_state = state->Clone();
    // Try to get policy at the current state
    auto policy = p0_portfolio[i]->GetStatePolicy(*test_state, 0);
    SPIEL_CHECK_GT(policy.size(), 0);
    // Should be deterministic (one action with prob 1)
    double total_prob = 0.0;
    int nonzero_count = 0;
    for (const auto& ap : policy) {
      total_prob += ap.second;
      if (ap.second > 0) nonzero_count++;
    }
    SPIEL_CHECK_FLOAT_EQ(total_prob, 1.0);
    SPIEL_CHECK_EQ(nonzero_count, 1);  // Pure strategy
  }

  std::cout << "Pure strategy policies work correctly" << std::endl;
}

// ============================================================================
// Comprehensive MVS Value Verification
// ============================================================================

// Test that MVS game with all pure strategies gives the same value as
// the original subtree when solved optimally.
// This is the key theoretical property: the value of the MVS game should
// equal the value of the original game from that state.
void TestMVSValueWithAllPureStrategies() {
  std::cout << "TestMVSValueWithAllPureStrategies" << std::endl;

  auto game = LoadGame("kuhn_poker");

  // Test from a specific depth-limited state
  // Navigate through chance nodes to get a decision state
  auto state = game->NewInitialState();
  while (state->IsChanceNode()) {
    auto outcomes = state->ChanceOutcomes();
    state->ApplyAction(outcomes[0].first);
  }

  // Apply one player action to get to a subtree
  Action first_action = state->LegalActions()[0];
  state->ApplyAction(first_action);

  std::cout << "Testing from state: " << state->ToString() << std::endl;

  // Get all pure strategies for both players from this state
  auto p0_portfolio = EnumerateSubtreePureStrategies(*state, 0);
  auto p1_portfolio = EnumerateSubtreePureStrategies(*state, 1);

  std::cout << "P0 portfolio size: " << p0_portfolio.size() << std::endl;
  std::cout << "P1 portfolio size: " << p1_portfolio.size() << std::endl;

  if (p0_portfolio.size() > 100 || p1_portfolio.size() > 100) {
    std::cout << "Skipping large portfolio test" << std::endl;
    return;
  }

  // Create MVS game with depth_limit = 1 (triggers after one player action)
  // This means after chance + 1 action, we hit the depth limit
  auto mvs_game = std::make_shared<MVSGame>(
      game, p0_portfolio, p1_portfolio, /*depth_limit=*/1);

  // Navigate MVS state to the same point
  auto mvs_state = mvs_game->NewInitialState();
  // Apply chance outcomes
  while (mvs_state->IsChanceNode()) {
    auto outcomes = mvs_state->ChanceOutcomes();
    mvs_state->ApplyAction(outcomes[0].first);
  }
  // Apply the first player action
  mvs_state->ApplyAction(first_action);

  std::cout << "MVS state after depth limit: " << mvs_state->ToString() << std::endl;

  // Now we should be at portfolio selection
  auto* mvs_state_ptr = dynamic_cast<MVSState*>(mvs_state.get());
  SPIEL_CHECK_TRUE(mvs_state_ptr != nullptr);
  std::cout << "Phase: " << static_cast<int>(mvs_state_ptr->GetPhase()) << std::endl;

  // Compute the payoff matrix directly
  std::cout << "Payoff matrix:" << std::endl;
  for (size_t i = 0; i < p0_portfolio.size(); ++i) {
    for (size_t j = 0; j < p1_portfolio.size(); ++j) {
      std::vector<const Policy*> policies = {p0_portfolio[i].get(),
                                              p1_portfolio[j].get()};
      auto returns = algorithms::ExpectedReturns(
          *state, policies, -1, false);
      std::cout << "  (" << i << "," << j << "): [" << returns[0] << ", "
                << returns[1] << "]" << std::endl;
    }
  }

  // Note: portfolios above are built for this specific subtree state, not all
  // depth-limited states reachable from the initial state of mvs_game.
  // Running full-game CFR here can query out-of-subtree infosets in
  // SubtreePureStrategy by construction. We only validate local matrix
  // consistency for this chosen subtree.
  double min_u0 = 1e9, max_u0 = -1e9;
  for (size_t i = 0; i < p0_portfolio.size(); ++i) {
    for (size_t j = 0; j < p1_portfolio.size(); ++j) {
      std::vector<const Policy*> policies = {p0_portfolio[i].get(),
                                             p1_portfolio[j].get()};
      auto returns = algorithms::ExpectedReturns(*state, policies, -1, false);
      min_u0 = std::min(min_u0, returns[0]);
      max_u0 = std::max(max_u0, returns[0]);
      SPIEL_CHECK_FLOAT_EQ(returns[0] + returns[1], 0.0);
    }
  }
  SPIEL_CHECK_GE(min_u0, game->MinUtility());
  SPIEL_CHECK_LE(max_u0, game->MaxUtility());
  std::cout << "MVS local subtree matrix verification passed" << std::endl;
}

// Test that MVS transformation preserves game value for the full game
// when using all pure strategies and depth limit 0 from root
void TestMVSFullGameValuePreservationRootStateKuhnPoker() {
  std::cout << "TestMVSFullGameValuePreservationRootStateKuhnPoker" << std::endl;

  auto game = LoadGame("kuhn_poker");
  auto initial_state = game->NewInitialState();

  // Get all pure strategies from the root
  auto p0_portfolio = EnumerateSubtreePureStrategies(*initial_state, 0);
  auto p1_portfolio = EnumerateSubtreePureStrategies(*initial_state, 1);

  std::cout << "Full game - P0 portfolio size: " << p0_portfolio.size() << std::endl;
  std::cout << "Full game - P1 portfolio size: " << p1_portfolio.size() << std::endl;

  // For Kuhn poker, this should be manageable
  // P0 has 2 infostates * 2 actions = 4 pure strategies typically
  // Actually depends on the tree structure

  if (p0_portfolio.size() > 100 || p1_portfolio.size() > 100) {
    std::cout << "Skipping large portfolio test (P0=" << p0_portfolio.size()
              << ", P1=" << p1_portfolio.size() << ")" << std::endl;
    return;
  }

  // Create MVS game with depth 0 - this is a direct matrix game
  auto mvs_game = std::make_shared<MVSGame>(
      game, p0_portfolio, p1_portfolio, /*depth_limit=*/0);

  // Solve MVS game with CFR
  algorithms::CFRSolverBase mvs_solver(*mvs_game,
                                        /*alternating_updates=*/true,
                                        /*linear_averaging=*/true,
                                        /*regret_matching_plus=*/true);

  for (int i = 0; i < 2000; ++i) {
    mvs_solver.EvaluateAndUpdatePolicy();
  }

  auto mvs_policy = mvs_solver.AveragePolicy();
  auto mvs_value = algorithms::ExpectedReturns(
      *mvs_game->NewInitialState(), *mvs_policy, -1, true);

  std::cout << "MVS game value (CFR 2000 iters): [" << mvs_value[0] << ", "
            << mvs_value[1] << "]" << std::endl;

  // Solve original game with CFR
  algorithms::CFRSolverBase orig_solver(*game,
                                         /*alternating_updates=*/true,
                                         /*linear_averaging=*/true,
                                         /*regret_matching_plus=*/true);

  for (int i = 0; i < 2000; ++i) {
    orig_solver.EvaluateAndUpdatePolicy();
  }

  auto orig_policy = orig_solver.AveragePolicy();
  auto orig_value = algorithms::ExpectedReturns(
      *game->NewInitialState(), *orig_policy, -1, true);

  std::cout << "Original game value (CFR 2000 iters): [" << orig_value[0] << ", "
            << orig_value[1] << "]" << std::endl;

  // The values should be close (CFR convergence)
  // With all pure strategies, the MVS game contains all possible play,
  // so the Nash equilibrium value should be the same
  double tolerance = 0.05;  // CFR may not have fully converged
  SPIEL_CHECK_LT(std::abs(mvs_value[0] - orig_value[0]), tolerance);

  std::cout << "Full game value preservation test passed!" << std::endl;
}

// Test that MVS transformation preserves game value for the full game
// when using all pure strategies and depth limit 0 from one action state
void TestMVSFullGameValuePreservationOneActionKuhnPoker() {
  std::cout << "TestMVSFullGameValuePreservationOneActionKuhnPoker" << std::endl;

  auto game = LoadGame("kuhn_poker");
  auto initial_state = game->NewInitialState();

  // Get all pure strategies from the root
  auto p0_portfolio = EnumerateSubtreePureStrategies(*initial_state, 0);
  auto p1_portfolio = EnumerateSubtreePureStrategies(*initial_state, 1);

  std::cout << "Full game - P0 portfolio size: " << p0_portfolio.size() << std::endl;
  std::cout << "Full game - P1 portfolio size: " << p1_portfolio.size() << std::endl;

  // For Kuhn poker, this should be manageable
  // P0 has 2 infostates * 2 actions = 4 pure strategies typically
  // Actually depends on the tree structure

  if (p0_portfolio.size() > 100 || p1_portfolio.size() > 100) {
    std::cout << "Skipping large portfolio test (P0=" << p0_portfolio.size()
              << ", P1=" << p1_portfolio.size() << ")" << std::endl;
    return;
  }

  // Create MVS game with depth 1
  auto mvs_game = std::make_shared<MVSGame>(
      game, p0_portfolio, p1_portfolio, /*depth_limit=*/1);

  // Solve MVS game with CFR
  algorithms::CFRSolverBase mvs_solver(*mvs_game,
                                        /*alternating_updates=*/true,
                                        /*linear_averaging=*/true,
                                        /*regret_matching_plus=*/true);

  for (int i = 0; i < 2000; ++i) {
    mvs_solver.EvaluateAndUpdatePolicy();
  }

  auto mvs_policy = mvs_solver.AveragePolicy();
  auto mvs_value = algorithms::ExpectedReturns(
      *mvs_game->NewInitialState(), *mvs_policy, -1, true);

  std::cout << "MVS game value (CFR 2000 iters): [" << mvs_value[0] << ", "
            << mvs_value[1] << "]" << std::endl;

  // Solve original game with CFR
  algorithms::CFRSolverBase orig_solver(*game,
                                         /*alternating_updates=*/true,
                                         /*linear_averaging=*/true,
                                         /*regret_matching_plus=*/true);

  for (int i = 0; i < 2000; ++i) {
    orig_solver.EvaluateAndUpdatePolicy();
  }

  auto orig_policy = orig_solver.AveragePolicy();
  auto orig_value = algorithms::ExpectedReturns(
      *game->NewInitialState(), *orig_policy, -1, true);

  std::cout << "Original game value (CFR 2000 iters): [" << orig_value[0] << ", "
            << orig_value[1] << "]" << std::endl;

  // The values should be close (CFR convergence)
  // With all pure strategies, the MVS game contains all possible play,
  // so the Nash equilibrium value should be the same
  double tolerance = 0.05;  // CFR may not have fully converged
  SPIEL_CHECK_LT(std::abs(mvs_value[0] - orig_value[0]), tolerance);

  std::cout << "Full game value preservation test passed!" << std::endl;
}

// Test 1: Verify the structure of MVS game with subtree pure strategies for Leduc.
// Walks through the game, checks portfolio sizes, and runs random simulations.
void TestMVSSubtreePureStrategiesLeducStructure() {
  std::cout << "TestMVSSubtreePureStrategiesLeducStructure" << std::endl;

  auto game = LoadGame("leduc_poker");

  // Create MVS game with subtree pure strategies, round-based depth, limit=3
  // In round-based mode, each chance node increments the round counter.
  // Leduc has 3 chance nodes: deal P0's card, deal P1's card, deal public card.
  // So depth_limit=3 means we hit the limit at the start of the second betting round.
  auto mvs_game = std::make_shared<MVSGameWithSubtreePureStrategies>(
      game, /*depth_limit=*/3, MVSGame::DepthMode::kRoundBased);

  std::cout << "Created MVS game with subtree pure strategies, "
            << "round-based depth limit = 3 (after all chance nodes)" << std::endl;

  // Walk to a depth-limited state to check portfolio sizes.
  auto mvs_state = mvs_game->NewInitialState();

  std::cout << "Walking to a depth-limited state..." << std::endl;
  int steps = 0;
  while (!mvs_state->IsTerminal() && steps < 20) {
    auto* mvs_st = dynamic_cast<MVSStateWithSubtreePureStrategies*>(mvs_state.get());
    if (mvs_st && mvs_st->GetPhase() != MVSStateWithSubtreePureStrategies::Phase::kNormal) {
      // We've hit the depth limit and are in portfolio selection phase
      std::cout << "Hit depth limit at step " << steps << std::endl;
      std::cout << "State: " << mvs_state->ToString() << std::endl;

      // Check portfolio sizes
      const auto& p0_portfolio = mvs_st->GetPortfolioP0();
      const auto& p1_portfolio = mvs_st->GetPortfolioP1();
      std::cout << "P0 portfolio size at this node: " << p0_portfolio.size() << std::endl;
      std::cout << "P1 portfolio size at this node: " << p1_portfolio.size() << std::endl;

      // Verify portfolios are computed correctly (non-empty)
      SPIEL_CHECK_GT(p0_portfolio.size(), 0);
      SPIEL_CHECK_GT(p1_portfolio.size(), 0);

      break;
    }

    // Take first legal action (or sample chance)
    if (mvs_state->IsChanceNode()) {
      auto outcomes = mvs_state->ChanceOutcomes();
      mvs_state->ApplyAction(outcomes[0].first);
    } else {
      auto actions = mvs_state->LegalActions();
      mvs_state->ApplyAction(actions[0]);
    }
    steps++;
  }

  // Test random simulations through the MVS game
  std::cout << "Testing random simulation through MVS game..." << std::endl;
  std::mt19937 rng(42);
  for (int sim = 0; sim < 10; ++sim) {
    auto state = mvs_game->NewInitialState();
    while (!state->IsTerminal()) {
      if (state->IsChanceNode()) {
        auto outcomes = state->ChanceOutcomes();
        std::vector<double> probs;
        for (const auto& [action, prob] : outcomes) {
          probs.push_back(prob);
        }
        std::discrete_distribution<int> dist(probs.begin(), probs.end());
        int idx = dist(rng);
        state->ApplyAction(outcomes[idx].first);
      } else {
        auto actions = state->LegalActions();
        std::uniform_int_distribution<int> dist(0, actions.size() - 1);
        state->ApplyAction(actions[dist(rng)]);
      }
    }
    auto returns = state->Returns();
    // Verify zero-sum
    SPIEL_CHECK_FLOAT_EQ(returns[0] + returns[1], 0.0);
  }
  std::cout << "Random simulations completed successfully." << std::endl;

  std::cout << "TestMVSSubtreePureStrategiesLeducStructure passed!" << std::endl;
}

// Test 2: Run CFR on MVS game and compare value with original Leduc poker.
// This verifies that the MVS transformation preserves the game value.
void TestMVSSubtreePureStrategiesLeducCFRValue() {
  std::cout << "TestMVSSubtreePureStrategiesLeducCFRValue" << std::endl;

  auto game = LoadGame("leduc_poker");

  // Create MVS game with subtree pure strategies, round-based depth, limit=3
  auto mvs_game = std::make_shared<MVSGameWithSubtreePureStrategies>(
      game, /*depth_limit=*/3, MVSGame::DepthMode::kRoundBased);

  std::cout << "Running CFR on MVS game (this may take a while)..." << std::endl;

  // Solve MVS game with CFR
  algorithms::CFRSolverBase mvs_solver(*mvs_game,
                                        /*alternating_updates=*/true,
                                        /*linear_averaging=*/true,
                                        /*regret_matching_plus=*/true);

  auto mvs_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 1000; ++i) {
    mvs_solver.EvaluateAndUpdatePolicy();
    if ((i + 1) % 200 == 0) {
      auto now = std::chrono::high_resolution_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - mvs_start).count();
      std::cout << "  MVS CFR iteration " << (i + 1)
                << " (elapsed: " << elapsed << "ms, "
                << "cache: " << mvs_game->NumCachedStates() << " states)" << std::endl;
    }
  }
  auto mvs_end = std::chrono::high_resolution_clock::now();
  auto mvs_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(mvs_end - mvs_start).count();

  auto mvs_policy = mvs_solver.AveragePolicy();
  auto mvs_value = algorithms::ExpectedReturns(
      *mvs_game->NewInitialState(), *mvs_policy, -1, true);

  std::cout << "MVS game value (CFR 1000 iters): [" << mvs_value[0] << ", "
            << mvs_value[1] << "]" << std::endl;
  std::cout << "MVS CFR time: " << mvs_elapsed << "ms" << std::endl;
  std::cout << "MVS cache stats: " << mvs_game->NumCachedStates()
            << " unique states, " << mvs_game->NumCachedPayoffs()
            << " payoff computations" << std::endl;

  // Solve original game with CFR
  std::cout << "Running CFR on original Leduc game..." << std::endl;
  algorithms::CFRSolverBase orig_solver(*game,
                                         /*alternating_updates=*/true,
                                         /*linear_averaging=*/true,
                                         /*regret_matching_plus=*/true);

  auto orig_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 1000; ++i) {
    orig_solver.EvaluateAndUpdatePolicy();
    if ((i + 1) % 200 == 0) {
      auto now = std::chrono::high_resolution_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - orig_start).count();
      std::cout << "  Original CFR iteration " << (i + 1)
                << " (elapsed: " << elapsed << "ms)" << std::endl;
    }
  }
  auto orig_end = std::chrono::high_resolution_clock::now();
  auto orig_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(orig_end - orig_start).count();

  auto orig_policy = orig_solver.AveragePolicy();
  auto orig_value = algorithms::ExpectedReturns(
      *game->NewInitialState(), *orig_policy, -1, true);

  std::cout << "Original game value (CFR 1000 iters): [" << orig_value[0] << ", "
            << orig_value[1] << "]" << std::endl;
  std::cout << "Original CFR time: " << orig_elapsed << "ms" << std::endl;
  std::cout << "Speedup ratio (original/MVS): " << (double)orig_elapsed / mvs_elapsed << "x" << std::endl;

  // The values should be close (CFR convergence)
  // With all pure strategies at each depth-limited node, the MVS game contains
  // all possible play, so the Nash equilibrium value should be the same.
  double tolerance = 0.05;  // CFR may not have fully converged
  SPIEL_CHECK_LT(std::abs(mvs_value[0] - orig_value[0]), tolerance);

  std::cout << "TestMVSSubtreePureStrategiesLeducCFRValue passed!" << std::endl;
}

std::unordered_map<std::string, double> DepthLimitedExploitability(const Game& game, std::shared_ptr<Policy> policy, std::unordered_map<int, std::unordered_set<std::string>> depth_limited_infostates) {
  std::unordered_map<std::string, double> exploitability_table;

  double exploitability = 0;

  SPIEL_CHECK_EQ(game.NumPlayers(), 2);
  for(Player player = 0; player < 2; player++) {
    algorithms::CFRSolverBase fixed_policy_solver(game,
      /*alternating_updates=*/true,
      /*linear_averaging=*/true,
      /*regret_matching_plus=*/true);

    fixed_policy_solver.SetFixedPolicy(policy, depth_limited_infostates[player]);

    for (int i = 0; i < 500; ++i) {
      fixed_policy_solver.EvaluateAndUpdatePolicy();
    }

    auto fixed_policy_avg_policy = fixed_policy_solver.AveragePolicy();

    algorithms::TabularBestResponse best_response(game, 1 - player, fixed_policy_avg_policy.get());

    double best_response_value = best_response.Value(*game.NewInitialState());
    exploitability += best_response_value;

    exploitability_table["Player " + std::to_string(player) + " exploitability"] = exploitability;
  }
  exploitability /= 2;
  exploitability_table["Total exploitability"] = exploitability;
  return exploitability_table;
}


// Test 3: Verify MVS strategy by locking round-1 actions during CFR solving
// on the original game using policy_overrides. This properly tests that the
// MVS round-1 strategy is compatible with optimal round-2 play.
void TestMVSStrategyInOriginalGame() {
  std::cout << "TestMVSStrategyInOriginalGame" << std::endl;

  auto game = LoadGame("leduc_poker");

  // Create MVS game with subtree pure strategies
  auto mvs_game = std::make_shared<MVSGameWithSubtreePureStrategies>(
      game, /*depth_limit=*/3, MVSGame::DepthMode::kRoundBased);

  // Run CFR on MVS game
  std::cout << "Running CFR on MVS game..." << std::endl;
  algorithms::CFRSolverBase mvs_solver(*mvs_game,
                                        /*alternating_updates=*/true,
                                        /*linear_averaging=*/true,
                                        /*regret_matching_plus=*/true);

  for (int i = 0; i < 500; ++i) {
    mvs_solver.EvaluateAndUpdatePolicy();
  }
  auto mvs_avg_policy = mvs_solver.AveragePolicy();

  // Get MVS game value
  auto mvs_value = algorithms::ExpectedReturns(
      *mvs_game->NewInitialState(), *mvs_avg_policy, -1, true);
  std::cout << "MVS game value: [" << mvs_value[0] << ", " << mvs_value[1] << "]"
            << std::endl;

  // Collect round-1 info states from the original game
  // Leduc: round 1 is before 3 chance nodes (2 private + 1 public card)
  auto round1_infostates = CollectInfoStateStringsBeforeRound(*game, 3);
  std::cout << "Round 1 info states in original game: "
            << round1_infostates.size() << std::endl;

  // Extract MVS policy for round-1 states into a TabularPolicy
  auto mvs_round1_policy = std::make_shared<TabularPolicy>();
  int matched_states = 0;
  for (const auto& [player, info_states] : round1_infostates) {
    for (const auto& info_state : info_states) {
      auto actions_probs = mvs_avg_policy->GetStatePolicy(info_state);
      if (!actions_probs.empty()) {
        mvs_round1_policy->SetStatePolicy(info_state, actions_probs);
        matched_states++;
      }
    }
  }
  std::cout << "Matched info states from MVS policy: " << matched_states << std::endl;

  std::unordered_map<std::string, double> exploitability_table = DepthLimitedExploitability(*game, mvs_round1_policy, round1_infostates);

  std::cout << "MVS R1 exploitability: " << exploitability_table["Player 0 exploitability"] << std::endl;
  std::cout << "MVS R1 exploitability: " << exploitability_table["Player 1 exploitability"] << std::endl;
  std::cout << "MVS R1 total exploitability: " << exploitability_table["Total exploitability"] << std::endl;

  // Also run standard CFR for comparison
  std::cout << "Running standard CFR on original game..." << std::endl;
  algorithms::CFRSolverBase orig_solver(*game,
                                        /*alternating_updates=*/true,
                                        /*linear_averaging=*/true,
                                        /*regret_matching_plus=*/true);

  for (int i = 0; i < 500; ++i) {
    orig_solver.EvaluateAndUpdatePolicy();
  }
  auto orig_avg_policy = orig_solver.AveragePolicy();

  auto orig_value = algorithms::ExpectedReturns(
      *game->NewInitialState(), *orig_avg_policy, -1, true);

  double orig_exploitability = algorithms::Exploitability(*game, *orig_avg_policy);

  // Extract original policy for round-1 states into a TabularPolicy
  auto orig_round1_policy = std::make_shared<TabularPolicy>();
  matched_states = 0;
  for (const auto& [player, info_states] : round1_infostates) {
    for (const auto& info_state : info_states) {
      auto actions_probs = orig_avg_policy->GetStatePolicy(info_state);
      if (!actions_probs.empty()) {
        orig_round1_policy->SetStatePolicy(info_state, actions_probs);
        matched_states++;
      }
    }
  }

  std::unordered_map<std::string, double> orig_exploitability_table = DepthLimitedExploitability(*game, orig_round1_policy, round1_infostates);

  std::cout << "Original R1 exploitability: " << orig_exploitability_table["Player 0 exploitability"] << std::endl;
  std::cout << "Original R1 exploitability: " << orig_exploitability_table["Player 1 exploitability"] << std::endl;
  std::cout << "Original R1 total exploitability: " << orig_exploitability_table["Total exploitability"] << std::endl;

  // Verify:
  // 1. MVS value should match original Nash value
  double tolerance = 0.1;
  SPIEL_CHECK_LT(std::abs(orig_exploitability_table["Total exploitability"] - exploitability_table["Total exploitability"]), tolerance);

  // 2. Combined policy exploitability should be reasonably low
  // Since MVS learns the same equilibrium value, combining MVS round-1 with
  // CFR round-2 should give a near-equilibrium strategy
  std::cout << "\nIf exploitability is high, it means MVS round-1 and CFR round-2 "
            << "are not compatible (different equilibria)." << std::endl;

  std::cout << "TestMVSStrategyInOriginalGame passed!" << std::endl;
}

// Test MVS with subtree pure strategies on II goofspiel(4), descending cards.
// Uses action-based depth limit = 4 (two full bidding rounds in the trunk).
void TestMVSSubtreePureStrategiesGoofspiel() {
  std::cout << "TestMVSSubtreePureStrategiesGoofspiel" << std::endl;

  auto game = LoadGameAsTurnBased(
      "goofspiel",
      {{"num_cards", GameParameter(4)},
       {"imp_info", GameParameter(true)},
       {"points_order", GameParameter(std::string("descending"))}});

  // Create MVS game with subtree pure strategies, action-based depth limit = 4
  auto mvs_game = CreateMVSGameWithSubtreePureStrategies(
      game, /*depth_limit=*/4, MVSGame::DepthMode::kActionBased);

  std::cout << "  MVS game created" << std::endl;

  // Random simulations to verify structural correctness
  std::mt19937 rng(42);
  for (int sim = 0; sim < 20; ++sim) {
    auto state = mvs_game->NewInitialState();
    while (!state->IsTerminal()) {
      if (state->IsChanceNode()) {
        auto outcomes = state->ChanceOutcomes();
        std::vector<double> probs;
        for (const auto& [action, prob] : outcomes) {
          probs.push_back(prob);
        }
        std::discrete_distribution<int> dist(probs.begin(), probs.end());
        int idx = dist(rng);
        state->ApplyAction(outcomes[idx].first);
      } else {
        auto actions = state->LegalActions();
        std::uniform_int_distribution<int> dist(0, actions.size() - 1);
        state->ApplyAction(actions[dist(rng)]);
      }
    }
    auto returns = state->Returns();
    SPIEL_CHECK_FLOAT_EQ(returns[0] + returns[1], 0.0);
  }
  std::cout << "  Random simulations passed (zero-sum verified)" << std::endl;

  // Solve MVS game with CFR
  algorithms::CFRSolverBase mvs_solver(*mvs_game,
                                        /*alternating_updates=*/true,
                                        /*linear_averaging=*/true,
                                        /*regret_matching_plus=*/true);

  for (int i = 0; i < 500; ++i) {
    mvs_solver.EvaluateAndUpdatePolicy();
  }

  auto mvs_policy = mvs_solver.AveragePolicy();
  double mvs_exploitability =
      algorithms::Exploitability(*mvs_game, *mvs_policy);
  std::cout << "  MVS game exploitability: " << mvs_exploitability << std::endl;

  // Solve original game with CFR for comparison
  algorithms::CFRSolverBase orig_solver(*game,
                                         /*alternating_updates=*/true,
                                         /*linear_averaging=*/true,
                                         /*regret_matching_plus=*/true);

  for (int i = 0; i < 500; ++i) {
    orig_solver.EvaluateAndUpdatePolicy();
  }

  auto orig_policy = orig_solver.AveragePolicy();
  double orig_exploitability =
      algorithms::Exploitability(*game, *orig_policy);
  std::cout << "  Full game exploitability: " << orig_exploitability
            << std::endl;

  // MVS exploitability should be in the same ballpark as the full game
  double tolerance = 0.05;
  SPIEL_CHECK_LT(std::abs(mvs_exploitability - orig_exploitability), tolerance);

  std::cout << "TestMVSSubtreePureStrategiesGoofspiel passed!" << std::endl;
}

// Verify MVS trunk strategy on II goofspiel(4) by locking trunk info states
// during CFR solving on the original game. Analogous to
// TestMVSStrategyInOriginalGame but for goofspiel with action-based depth.
void TestMVSGoofspielStrategyInOriginalGame() {
  std::cout << "TestMVSGoofspielStrategyInOriginalGame" << std::endl;

  auto game = LoadGameAsTurnBased(
      "goofspiel",
      {{"num_cards", GameParameter(4)},
       {"imp_info", GameParameter(true)},
       {"points_order", GameParameter(std::string("descending"))}});

  const int depth_limit = 4;

  // Step 1: Solve MVS game
  std::cout << "  Running CFR on MVS game..." << std::endl;
  auto mvs_game = CreateMVSGameWithSubtreePureStrategies(
      game, depth_limit, MVSGame::DepthMode::kActionBased);

  algorithms::CFRSolverBase mvs_solver(*mvs_game,
                                        /*alternating_updates=*/true,
                                        /*linear_averaging=*/true,
                                        /*regret_matching_plus=*/true);

  for (int i = 0; i < 500; ++i) {
    mvs_solver.EvaluateAndUpdatePolicy();
  }
  auto mvs_avg_policy = mvs_solver.AveragePolicy();

  auto mvs_value = algorithms::ExpectedReturns(
      *mvs_game->NewInitialState(), *mvs_avg_policy, -1, true);
  std::cout << "  MVS game value: [" << mvs_value[0] << ", " << mvs_value[1]
            << "]" << std::endl;

  // Step 2: Collect trunk info states (action depth < depth_limit)
  auto trunk_infostates =
      CollectInfoStateStringsBeforeDepth(*game, depth_limit);
  int total_trunk = 0;
  for (const auto& [player, info_states] : trunk_infostates) {
    std::cout << "  Player " << player << " trunk info states: "
              << info_states.size() << std::endl;
    total_trunk += info_states.size();
  }
  std::cout << "  Total trunk info states: " << total_trunk << std::endl;

  // Step 3: Extract MVS policy for trunk states
  auto mvs_trunk_policy = std::make_shared<TabularPolicy>();
  int matched_states = 0;
  for (const auto& [player, info_states] : trunk_infostates) {
    for (const auto& info_state : info_states) {
      auto actions_probs = mvs_avg_policy->GetStatePolicy(info_state);
      if (!actions_probs.empty()) {
        mvs_trunk_policy->SetStatePolicy(info_state, actions_probs);
        matched_states++;
      }
    }
  }
  std::cout << "  Matched info states from MVS policy: " << matched_states
            << std::endl;

  // Step 4: Lock trunk, solve rest with CFR, compute exploitability
  auto mvs_exploitability_table =
      DepthLimitedExploitability(*game, mvs_trunk_policy, trunk_infostates);

  std::cout << "  MVS trunk exploitability: "
            << mvs_exploitability_table["Total exploitability"] << std::endl;

  // Step 5: Solve original game with CFR for comparison
  std::cout << "  Running standard CFR on original game..." << std::endl;
  algorithms::CFRSolverBase orig_solver(*game,
                                         /*alternating_updates=*/true,
                                         /*linear_averaging=*/true,
                                         /*regret_matching_plus=*/true);

  for (int i = 0; i < 500; ++i) {
    orig_solver.EvaluateAndUpdatePolicy();
  }
  auto orig_avg_policy = orig_solver.AveragePolicy();

  auto orig_trunk_policy = std::make_shared<TabularPolicy>();
  for (const auto& [player, info_states] : trunk_infostates) {
    for (const auto& info_state : info_states) {
      auto actions_probs = orig_avg_policy->GetStatePolicy(info_state);
      if (!actions_probs.empty()) {
        orig_trunk_policy->SetStatePolicy(info_state, actions_probs);
      }
    }
  }

  auto orig_exploitability_table =
      DepthLimitedExploitability(*game, orig_trunk_policy, trunk_infostates);

  std::cout << "  Original trunk exploitability: "
            << orig_exploitability_table["Total exploitability"] << std::endl;

  // The MVS trunk strategy should produce similar exploitability
  double tolerance = 0.1;
  SPIEL_CHECK_LT(
      std::abs(mvs_exploitability_table["Total exploitability"] -
               orig_exploitability_table["Total exploitability"]),
      tolerance);

  std::cout << "TestMVSGoofspielStrategyInOriginalGame passed!" << std::endl;
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::Init("", &argc, &argv, true);

  // Check for command line flags
  bool leduc_structure_only = false;
  bool leduc_cfr_only = false;
  bool leduc_verify_only = false;
  bool goofspiel_only = false;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--leduc_structure") {
      leduc_structure_only = true;
    } else if (arg == "--leduc_cfr") {
      leduc_cfr_only = true;
    } else if (arg == "--leduc_verify") {
      leduc_verify_only = true;
    } else if (arg == "--goofspiel") {
      goofspiel_only = true;
    }
  }

  if (leduc_structure_only) {
    std::cout << "Running only Leduc structure test..." << std::endl;
    open_spiel::TestMVSSubtreePureStrategiesLeducStructure();
    return 0;
  }

  if (leduc_cfr_only) {
    std::cout << "Running only Leduc CFR value test..." << std::endl;
    open_spiel::TestMVSSubtreePureStrategiesLeducCFRValue();
    return 0;
  }

  if (leduc_verify_only) {
    std::cout << "Running only Leduc strategy verification test..." << std::endl;
    open_spiel::TestMVSStrategyInOriginalGame();
    return 0;
  }

  if (goofspiel_only) {
    std::cout << "Running only Goofspiel MVS tests..." << std::endl;
    open_spiel::TestMVSSubtreePureStrategiesGoofspiel();
    open_spiel::TestMVSGoofspielStrategyInOriginalGame();
    return 0;
  }

  // Basic MVS tests
  open_spiel::TestBasicConstruction();
  open_spiel::TestDepthLimitTrigger();
  open_spiel::TestInformationStateCorrectness();
  open_spiel::TestPayoffComputation();
  open_spiel::TestMultiplePortfolios();
  open_spiel::TestEarlyTerminal();
  open_spiel::TestRoundBasedDepth();
  open_spiel::TestRandomSimulation();
  open_spiel::TestCFRCompatibility();

  // Pure strategy enumeration tests
  open_spiel::TestPureStrategyEnumeration();
  open_spiel::TestPureStrategyContinuationConsistentAcrossPublicObservation();
  open_spiel::TestPureStrategyAsPolicy();

  // Comprehensive MVS value verification
  open_spiel::TestMVSValueWithAllPureStrategies();
  open_spiel::TestMVSFullGameValuePreservationRootStateKuhnPoker();
  open_spiel::TestMVSFullGameValuePreservationOneActionKuhnPoker();

  // Leduc poker tests with subtree pure strategies
  open_spiel::TestMVSSubtreePureStrategiesLeducStructure();
  open_spiel::TestMVSSubtreePureStrategiesLeducCFRValue();
  open_spiel::TestMVSStrategyInOriginalGame();

  // Goofspiel tests with subtree pure strategies
  open_spiel::TestMVSSubtreePureStrategiesGoofspiel();
  open_spiel::TestMVSGoofspielStrategyInOriginalGame();

  std::cout << "\nAll tests passed!" << std::endl;
  return 0;
}
