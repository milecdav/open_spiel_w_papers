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
#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"

ABSL_FLAG(int, iterations, 1000, "Number of CFR iterations to run");
ABSL_FLAG(bool, save_states, false, "Whether to save states");
ABSL_FLAG(bool, mccfr, false, "Whether to run MCCFR");
ABSL_FLAG(bool, cfr, false, "Whether to run Vanilla CFR");
ABSL_FLAG(bool, rnr, false, "Whether to run RNR");
ABSL_FLAG(bool, secfr, false, "Whether to run SECFRSolver");
ABSL_FLAG(int, target_player, 0, "Target player for RNR");


#include "algorithms/cfr.h"
#include "algorithms/tabular_exploitability.h"
#include "algorithms/best_response.h"
#include "algorithms/external_sampling_mccfr.h"
#include "algorithms/expected_returns.h"
#include "algorithms/rnr.h"
#include "secfr.h"
#if OPEN_SPIEL_BUILD_WITH_ORTOOLS
  #include "algorithms/ortools/sequence_form_lp.h"
#endif
#include "policy.h"

#include <iostream>


namespace open_spiel {
namespace papers_with_code {
namespace {

  double GetGameValueForPlayer(std::shared_ptr<const Game> game, int player_index) {
    #if OPEN_SPIEL_BUILD_WITH_ORTOOLS
      algorithms::ortools::SequenceFormLpSpecification specification(*game);
      specification.SpecifyLinearProgram(player_index);
      return specification.Solve();
    #else
      if(game->ToString() == "leduc_poker()") {
        std::cout << "Warning: OR-Tools is not built, using known game value for Leduc Poker\n";
        if(player_index == 0) {
          return -0.0856064;
        } else {
          return 0.0856064;
        }
      } else {
        throw std::runtime_error("Game value not supported without OR-Tools for game: " + game->ToString());
      }
    #endif
  }

  double AdjustedBRValue(std::shared_ptr<const Game> game, int player_index, std::shared_ptr<Policy> average_policy) {
    algorithms::TabularBestResponse best_response(*game, player_index, average_policy.get());
    double best_response_value = best_response.Value(*game->NewInitialState());
    return best_response_value - GetGameValueForPlayer(game, player_index);
  }

  void RunVanillaCFR(int iterations, bool save_states) {
    std::shared_ptr<const Game> game = LoadGame("leduc_poker");
    algorithms::CFRSolverBase solver(*game, false, true, true, save_states);
    for (int i = 0; i < iterations; i++) {
      solver.EvaluateAndUpdatePolicy();
    }
    std::shared_ptr<Policy> average_policy = solver.AveragePolicy();
    double exploitability = algorithms::Exploitability(*game, *average_policy);
    std::cout << "Exploitability: " << exploitability << "\n";
    for(int player_index = 0; player_index < game->NumPlayers(); player_index++) {
      algorithms::TabularBestResponse best_response(*game, player_index, average_policy.get());
      double best_response_value = best_response.Value(*game->NewInitialState());
      std::cout << "Best response value for player " << player_index << ": " << best_response_value << "\n";
    }
    std::cout << "Game value for player 0: " << GetGameValueForPlayer(game, 0) << "\n";
    std::cout << "Game value for player 1: " << GetGameValueForPlayer(game, 1) << "\n";
  }

  void RunMCCFR(int iterations) {
    std::shared_ptr<const Game> game = LoadGame("leduc_poker");
    algorithms::ExternalSamplingMCCFRSolver solver(*game);
    for (int i = 0; i < iterations; i++) {
      solver.RunIteration();
    }
    std::shared_ptr<Policy> average_policy = solver.AveragePolicy();
    double exploitability = algorithms::Exploitability(*game, *average_policy);
    std::cout << "Exploitability: " << exploitability << "\n";
    for(int player_index = 0; player_index < game->NumPlayers(); player_index++) {
      algorithms::TabularBestResponse best_response(*game, player_index, average_policy.get());
      double best_response_value = best_response.Value(*game->NewInitialState());
      std::cout << "Best response value for player " << player_index << ": " << best_response_value << "\n";
    }
  }

  void RunRNR(std::shared_ptr<const Game> game, int iterations, double p, int target_player, std::shared_ptr<Policy> fixed_opponent_policy) {
    std::cout << "Running RNR with p = " << p << std::endl;
    algorithms::RNRSolver solver(*game, target_player, fixed_opponent_policy.get(), p, false, true, true, true);
    for (int i = 0; i < iterations; i++) {
      solver.EvaluateAndUpdatePolicy();
    }
    std::shared_ptr<Policy> average_policy = solver.AveragePolicy();
    double adjusted_br_value = AdjustedBRValue(game, 1 - target_player, average_policy);
    std::cout << "Adjusted best response value for player " << 1 - target_player << ": " << adjusted_br_value << "\n";
    std::cout << "Expected return value for player " << target_player << ": " << algorithms::ExpectedReturns(*game->NewInitialState(), {average_policy.get(), fixed_opponent_policy.get()}, -1)[target_player] << "\n";
    std::cout << "--------------------------------" << std::endl;
  }

  void RunSECFRSolver(std::shared_ptr<const Game> game, int iterations, double p, int target_player, std::shared_ptr<Policy> fixed_opponent_policy) {
    std::cout << "Running SECFRSolver with p = " << p << std::endl;
    SECFRSolver solver(*game, fixed_opponent_policy.get(), p, false, true, true, true);
    for (int i = 0; i < iterations; i++) {
      solver.EvaluateAndUpdatePolicy();
    }
    std::shared_ptr<Policy> average_policy = solver.AveragePolicy();
    double adjusted_br_value = AdjustedBRValue(game, 1 - target_player, average_policy);
    std::cout << "Adjusted best response value for player " << 1 - target_player << ": " << adjusted_br_value << "\n";
    std::cout << "Expected return value for player " << target_player << ": " << algorithms::ExpectedReturns(*game->NewInitialState(), {average_policy.get(), fixed_opponent_policy.get()}, -1)[target_player] << "\n";
    std::cout << "--------------------------------" << std::endl;
  }

  void RunSweep(std::shared_ptr<const Game> game, int iterations, int target_player, std::shared_ptr<Policy> fixed_opponent_policy, std::function<void(std::shared_ptr<const Game>, int, double, int, std::shared_ptr<Policy>)> run_function) {
    for(double p = 0.0; p <= 1.0; p += 0.1) {
      run_function(game, iterations, p, target_player, fixed_opponent_policy);
    }
  }

  void RunSweep(int iterations, int target_player, std::function<void(std::shared_ptr<const Game>, int, double, int, std::shared_ptr<Policy>)> run_function) {
    std::shared_ptr<const Game> game = LoadGame("leduc_poker");
    std::shared_ptr<Policy> fixed_opponent_policy = std::make_shared<TabularPolicy>(*game);
    RunSweep(game, iterations, target_player, fixed_opponent_policy, run_function);
  }
} // unnamed namespace
} // namespace papers_with_code
} // namespace open_spiel

int main(int argc, char **argv) {

  std::vector<char *> positional_args = absl::ParseCommandLine(argc, argv);  
  bool run_mccfr = absl::GetFlag(FLAGS_mccfr);
  bool run_vanilla_cfr = absl::GetFlag(FLAGS_cfr);
  bool run_rnr = absl::GetFlag(FLAGS_rnr);
  int target_player = absl::GetFlag(FLAGS_target_player);
  int iterations = absl::GetFlag(FLAGS_iterations);
  bool save_states = absl::GetFlag(FLAGS_save_states);
  bool run_secfr = absl::GetFlag(FLAGS_secfr);


  if(run_mccfr) {
    std::cout << "Running MCCFR with " << iterations << " iterations" << std::endl;
    open_spiel::papers_with_code::RunMCCFR(iterations);
  }
  if(run_vanilla_cfr) {
    std::cout << "Running Vanilla CFR with " << iterations << " iterations" << std::endl;
    open_spiel::papers_with_code::RunVanillaCFR(iterations, save_states);
  }
  if(run_rnr) {
    std::cout << "Running RNR with " << iterations << " iterations and target player " << target_player << std::endl;
    open_spiel::papers_with_code::RunSweep(iterations, target_player, open_spiel::papers_with_code::RunRNR);
  }
  if(run_secfr) {
    std::cout << "Running SECFRSolver with " << iterations << " iterations and target player " << target_player << std::endl;
    open_spiel::papers_with_code::RunSweep(iterations, target_player, open_spiel::papers_with_code::RunSECFRSolver);
  }

  return 0;
}