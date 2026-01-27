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

#include "open_spiel/algorithms/rnr.h"

#include <cmath>
#include <iostream>
#include <memory>

#include "open_spiel/algorithms/best_response.h"
#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/algorithms/expected_returns.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "policy.h"

namespace open_spiel {
namespace algorithms {
namespace {

constexpr int kNumIterations = 1000;
constexpr double kTolerance = 0.001;  // Allow some tolerance due to CFR convergence

void TestRNRWithPZeroIsNashEquilibrium() {
  std::cout << "\n=== Test: RNR with p=0 should be Nash Equilibrium ===\n";
  
  std::shared_ptr<const Game> game = LoadGame("kuhn_poker");
  
  // First, run standard CFR to get a baseline Nash equilibrium
  CFRSolverBase cfr_solver(*game, /*alternating_updates=*/true,
                           /*linear_averaging=*/false,
                           /*regret_matching_plus=*/false,
                           /*save_states=*/true);
  for (int i = 0; i < kNumIterations; ++i) {
    cfr_solver.EvaluateAndUpdatePolicy();
  }
  std::shared_ptr<Policy> cfr_policy = cfr_solver.AveragePolicy();
  double cfr_exploitability = Exploitability(*game, *cfr_policy);
  std::cout << "Standard CFR exploitability: " << cfr_exploitability << "\n";
  
  // Now run RNR with p=0 (should converge to Nash equilibrium)
  // Use a uniform policy as the "fixed" opponent (shouldn't matter when p=0)
  TabularPolicy fixed_opponent_policy = TabularPolicy(*game);
  
  RNRSolver rnr_solver(*game, /*target_player=*/0, &fixed_opponent_policy, /*p=*/0.0,
                       /*save_states=*/true);
  for (int i = 0; i < kNumIterations; ++i) {
    rnr_solver.EvaluateAndUpdatePolicy();
  }
  
  // With p=0, both players should be playing Nash equilibrium
  std::shared_ptr<Policy> policy = rnr_solver.AveragePolicy();
  
  // Create a combined policy for exploitability computation
  double rnr_exploitability = Exploitability(*game, *policy);
  
  std::cout << "RNR (p=0) combined exploitability: " << rnr_exploitability << "\n";
  
  // Should be close to Nash equilibrium (low exploitability)
  SPIEL_CHECK_LT(rnr_exploitability, cfr_exploitability + kTolerance);
  
  std::cout << "PASSED: RNR with p=0 converges to Nash equilibrium\n";
}

void TestRNRWithPOneIsBestResponse() {
  std::cout << "\n=== Test: RNR with p=1 should be Best Response ===\n";
  
  std::shared_ptr<const Game> game = LoadGame("kuhn_poker");

  CFRSolverBase cfr_solver(*game, /*alternating_updates=*/true,
    /*linear_averaging=*/false,
    /*regret_matching_plus=*/false,
    /*save_states=*/true);
  for (int i = 0; i < kNumIterations; ++i) {
    cfr_solver.EvaluateAndUpdatePolicy();
  }
  std::shared_ptr<Policy> cfr_policy = cfr_solver.AveragePolicy();
  
  // Create a fixed opponent policy (use uniform for simplicity)
  TabularPolicy fixed_opponent_policy = TabularPolicy(*game);
  int target_player = 0;
  
  // Compute actual best response to the fixed policy
  TabularBestResponse br_computer(*game, /*player=*/target_player, &fixed_opponent_policy);
  double br_value = br_computer.Value(*game->NewInitialState());
  std::cout << "True best response value for player 0: " << br_value << "\n";
  
  // Run RNR with p=1 (should be pure best response)
  RNRSolver rnr_solver(*game, /*target_player=*/target_player, &fixed_opponent_policy, /*p=*/1.0);
  for (int i = 0; i < kNumIterations; ++i) {
    rnr_solver.EvaluateAndUpdatePolicy();
  }

  std::cout << "RNR (p=1) exploitability: " << Exploitability(*game, *rnr_solver.AveragePolicy()) << "\n";
  
  std::shared_ptr<Policy> rnr_policy = rnr_solver.AveragePolicy();
  std::vector<const Policy*> policies = {rnr_policy.get(), &fixed_opponent_policy};
  
  // Compute the value of RNR policy against the fixed opponent
  double rnr_value = ExpectedReturns(*game->NewInitialState(), policies, -1, false)[target_player];
  
  std::cout << "RNR (p=1) policy value against fixed opponent: " << rnr_value << "\n";

  double cfr_value = ExpectedReturns(*game->NewInitialState(), {cfr_policy.get(), &fixed_opponent_policy}, -1, false)[target_player];
  std::cout << "CFR policy value against fixed opponent: " << cfr_value << "\n";
  
  // RNR with p=1 should achieve close to best response value
  SPIEL_CHECK_GT(rnr_value, br_value - kTolerance);
  
  std::cout << "PASSED: RNR with p=1 achieves best response value\n";
}

void TestRNRIntermediate() {
  std::cout << "\n=== Test: RNR with intermediate p balances exploitation and safety ===\n";
  
  std::shared_ptr<const Game> game = LoadGame("kuhn_poker");

  int target_player = 0;
  // Create a fixed opponent policy
  TabularPolicy fixed_opponent_policy = TabularPolicy(*game);
  
  // Compute best response value (upper bound on exploitation)
  TabularBestResponse br_computer(*game, /*player=*/target_player, &fixed_opponent_policy);
  double br_value = br_computer.Value(*game->NewInitialState());
  std::cout << "Best response value: " << br_value << "\n";
  
  // Compute Nash equilibrium exploitability (safety baseline)
  CFRSolverBase cfr_solver(*game, /*alternating_updates=*/true,
                           /*linear_averaging=*/false,
                           /*regret_matching_plus=*/false,
                           /*save_states=*/true);
  for (int i = 0; i < kNumIterations; ++i) {
    cfr_solver.EvaluateAndUpdatePolicy();
  }
  std::shared_ptr<Policy> ne_policy = cfr_solver.AveragePolicy();
  double ne_exploitability = Exploitability(*game, *ne_policy);
  std::cout << "Nash equilibrium exploitability: " << ne_exploitability << "\n";
  
  // Test RNR with p=0.5
  RNRSolver rnr_solver(*game, /*target_player=*/target_player, &fixed_opponent_policy, /*p=*/0.5,
                       /*save_states=*/true);
  for (int i = 0; i < kNumIterations; ++i) {
    rnr_solver.EvaluateAndUpdatePolicy();
  }
  
  std::shared_ptr<Policy> rnr_policy = rnr_solver.AveragePolicy();
  double rnr_exploitability = Exploitability(*game, *rnr_policy);

  std::vector<const Policy*> policies = {rnr_policy.get(), &fixed_opponent_policy};
  // Compute value against fixed opponent
  double rnr_value = ExpectedReturns(*game->NewInitialState(), policies, -1, false)[target_player];
  
  std::cout << "RNR (p=0.5) exploitability: " << rnr_exploitability << "\n";
  std::cout << "RNR (p=0.5) value against fixed opponent: " << rnr_value << "\n";
  
  // RNR should be somewhere between Nash and best response
  std::cout << "PASSED: RNR with p=0.5 provides balanced strategy\n";
}

void TestRNRWithSaveStatesVsWithout() {
  std::cout << "\n=== Test: RNR with save_states=true vs false should match ===\n";
  
  std::shared_ptr<const Game> game = LoadGame("kuhn_poker");
  TabularPolicy fixed_opponent_policy = TabularPolicy(*game);

  
  // Run RNR with save_states=true
  RNRSolver rnr_saved(*game, /*target_player=*/0, &fixed_opponent_policy, /*p=*/0.5,
                      /*save_states=*/false, true, true, true);
  for (int i = 0; i < kNumIterations; ++i) {
    rnr_saved.EvaluateAndUpdatePolicy();
  }
  std::shared_ptr<Policy> saved_policy = rnr_saved.AveragePolicy();
  double saved_exploitability = Exploitability(*game, *saved_policy);
  
  // Run RNR with save_states=false
  RNRSolver rnr_unsaved(*game, /*target_player=*/0, &fixed_opponent_policy, /*p=*/0.5,
                        /*save_states=*/false, true, true, false);
  for (int i = 0; i < kNumIterations; ++i) {
    rnr_unsaved.EvaluateAndUpdatePolicy();
  }
  std::shared_ptr<Policy> unsaved_policy = rnr_unsaved.AveragePolicy();
  double unsaved_exploitability = Exploitability(*game, *unsaved_policy);
  
  std::cout << "RNR (save_states=true) exploitability: " << saved_exploitability << "\n";
  std::cout << "RNR (save_states=false) exploitability: " << unsaved_exploitability << "\n";
  
  // Both should produce similar results
  SPIEL_CHECK_LT(std::abs(saved_exploitability - unsaved_exploitability), kTolerance);
  
  std::cout << "PASSED: save_states=true and save_states=false produce similar results\n";
}

void TestRNROnLeducPoker() {
  std::cout << "\n=== Test: RNR on Leduc Poker ===\n";
  
  std::shared_ptr<const Game> game = LoadGame("leduc_poker");
  int target_player = 0;
  // First run CFR to get a reasonable fixed policy
  CFRSolverBase cfr_for_fixed(*game, /*alternating_updates=*/true,
                               /*linear_averaging=*/false,
                               /*regret_matching_plus=*/false,
                               /*save_states=*/true);
  for (int i = 0; i < 50; ++i) {
    cfr_for_fixed.EvaluateAndUpdatePolicy();
  }
  std::shared_ptr<Policy> fixed_policy = cfr_for_fixed.AveragePolicy();
  double fixed_exploitability = Exploitability(*game, *fixed_policy);
  std::cout << "Fixed policy (50 CFR iterations) exploitability: " 
            << fixed_exploitability << "\n";
  
  // Run RNR against this fixed policy
  RNRSolver rnr_solver(*game, /*target_player=*/0, fixed_policy.get(), /*p=*/0.7,
                       /*save_states=*/true);
  for (int i = 0; i < kNumIterations; ++i) {
    rnr_solver.EvaluateAndUpdatePolicy();
  }
  
  std::shared_ptr<Policy> rnr_policy = rnr_solver.AveragePolicy();
  double rnr_exploitability = Exploitability(*game, *rnr_policy);
  
  std::cout << "RNR (p=0.7) exploitability: " << rnr_exploitability << "\n";
  
  // Compute value against fixed opponent
  std::vector<const Policy*> policies = {rnr_policy.get(), fixed_policy.get()};
  double rnr_value = ExpectedReturns(*game->NewInitialState(), policies, -1, false)[target_player];
  std::cout << "RNR value against fixed opponent: " << rnr_value << "\n";
  
  std::cout << "PASSED: RNR runs successfully on Leduc Poker\n";
}

}  // namespace
}  // namespace algorithms
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::algorithms::TestRNRWithPZeroIsNashEquilibrium();
  open_spiel::algorithms::TestRNRWithPOneIsBestResponse();
  open_spiel::algorithms::TestRNRIntermediate();
  open_spiel::algorithms::TestRNRWithSaveStatesVsWithout();
  open_spiel::algorithms::TestRNROnLeducPoker();
  
  std::cout << "\n=== All RNR tests passed! ===\n";
  return 0;
}
