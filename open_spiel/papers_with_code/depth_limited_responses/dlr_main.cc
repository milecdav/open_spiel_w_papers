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

#include "algorithms/cfr.h"
#include "algorithms/tabular_exploitability.h"
#include "algorithms/best_response.h"
#include "algorithms/external_sampling_mccfr.h"
#if OPEN_SPIEL_BUILD_WITH_ORTOOLS
  #include "algorithms/ortools/sequence_form_lp.h"
#endif
#include "policy.h"

#include <iostream>


namespace open_spiel {
namespace papers_with_code {
namespace {

  double GetGameValueForPlayer(std::shared_ptr<const Game> game, int player_index) {
    std::cout << game->ToString();
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
} // unnamed namespace
} // namespace papers_with_code
} // namespace open_spiel

int main(int argc, char **argv) {

  std::vector<char *> positional_args = absl::ParseCommandLine(argc, argv);  
  bool run_mccfr = absl::GetFlag(FLAGS_mccfr);
  bool run_vanilla_cfr = absl::GetFlag(FLAGS_cfr);
  int iterations = absl::GetFlag(FLAGS_iterations);
  bool save_states = absl::GetFlag(FLAGS_save_states);


  if(run_mccfr) {
    std::cout << "Running MCCFR with " << iterations << " iterations" << std::endl;
    open_spiel::papers_with_code::RunMCCFR(iterations);
  }
  if(run_vanilla_cfr) {
    std::cout << "Running Vanilla CFR with " << iterations << " iterations" << std::endl;
    open_spiel::papers_with_code::RunVanillaCFR(iterations, save_states);
  }

  return 0;
}