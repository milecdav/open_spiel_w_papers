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
ABSL_FLAG(std::string, game, "leduc_poker", "Game to run");


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
#include <fstream>

std::string RESULTS_DIR = "results/dlr/";

struct SweepResult {
  double p;
  double expl;
  double gain;
};

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

  void RunVanillaCFR(std::shared_ptr<const Game> game, int iterations, bool save_states) {
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

  void RunMCCFR(std::shared_ptr<const Game> game, int iterations) {
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

  std::pair<double, double> RunRNR(std::shared_ptr<const Game> game, int iterations, double p, int target_player, std::shared_ptr<Policy> fixed_opponent_policy) {
    std::cout << "Running RNR with p = " << p << std::endl;
    algorithms::RNRSolver solver(*game, target_player, fixed_opponent_policy.get(), p, false, true, true, true);
    for (int i = 0; i < iterations; i++) {
      solver.EvaluateAndUpdatePolicy();
    }
    std::shared_ptr<Policy> average_policy = solver.AveragePolicy();
    double adjusted_br_value = AdjustedBRValue(game, 1 - target_player, average_policy);
    double expected_return = algorithms::ExpectedReturns(*game->NewInitialState(), {average_policy.get(), fixed_opponent_policy.get()}, -1)[target_player];
    std::cout << "Adjusted best response value for player " << 1 - target_player << ": " << adjusted_br_value << "\n";
    std::cout << "Expected return value for player " << target_player << ": " << algorithms::ExpectedReturns(*game->NewInitialState(), {average_policy.get(), fixed_opponent_policy.get()}, -1)[target_player] << "\n";
    std::cout << "--------------------------------" << std::endl;

    return std::make_pair(adjusted_br_value, expected_return);
  }

  std::pair<double, double> RunSECFRSolver(std::shared_ptr<const Game> game, int iterations, double p, int target_player, std::shared_ptr<Policy> fixed_opponent_policy) {
    std::cout << "Running SECFRSolver with p = " << p << std::endl;
    SECFRSolver solver(*game, fixed_opponent_policy.get(), p, false, true, true, true);
    for (int i = 0; i < iterations; i++) {
      solver.EvaluateAndUpdatePolicy();
    }
    std::shared_ptr<Policy> average_policy = solver.AveragePolicy();
    double adjusted_br_value = AdjustedBRValue(game, 1 - target_player, average_policy);
    double expected_return = algorithms::ExpectedReturns(*game->NewInitialState(), {average_policy.get(), fixed_opponent_policy.get()}, -1)[target_player];
    std::cout << "Adjusted best response value for player " << 1 - target_player << ": " << adjusted_br_value << "\n";
    std::cout << "Expected return value for player " << target_player << ": " << algorithms::ExpectedReturns(*game->NewInitialState(), {average_policy.get(), fixed_opponent_policy.get()}, -1)[target_player] << "\n";
    std::cout << "--------------------------------" << std::endl;
    return std::make_pair(adjusted_br_value, expected_return);
  }

  std::vector<SweepResult> RunSweep(std::shared_ptr<const Game> game, int iterations, int target_player, std::shared_ptr<Policy> fixed_opponent_policy, std::function<std::pair<double, double>(std::shared_ptr<const Game>, int, double, int, std::shared_ptr<Policy>)> run_function) {
    std::vector<SweepResult> results;
    for(double p = 0.0; p <= 1.0; p += 0.1) {
      std::pair<double, double> result = run_function(game, iterations, p, target_player, fixed_opponent_policy);
      results.push_back({p, result.first, result.second});
    }
    return results; 
  }

  std::vector<SweepResult> RunSweep(std::shared_ptr<const Game> game, int iterations, int target_player, std::function<std::pair<double, double>(std::shared_ptr<const Game>, int, double, int, std::shared_ptr<Policy>)> run_function) {
    std::shared_ptr<Policy> fixed_opponent_policy = std::make_shared<TabularPolicy>(*game);
    return RunSweep(game, iterations, target_player, fixed_opponent_policy, run_function);
  }

  void SaveSweepResults(const std::vector<SweepResult>& results, std::string algorithm_name, std::string filename) {
    std::ofstream file(filename);
    file << "algorithm_name," << algorithm_name << "\n";
    file << "p,expl,gain\n";
    for(const SweepResult& result : results) {
      file << result.p << "," << result.expl << "," << result.gain << "\n";
    }
    file.close();
  }

  std::shared_ptr<const Game> GetGameFromFlag(std::string flag) {
    std::vector<std::string> implemented_games = {
      "leduc_poker",
      "goofspielN",
      "liars_dice(S,D)"
    };
    if(flag == "leduc_poker") {
      return LoadGame("leduc_poker");
    }
    if(flag.substr(0, 9) == "goofspiel" && flag.length() > 9) {
      std::string num_cards =
      flag.substr(9); // Extract number after "goofspiel"
      return LoadGame("turn_based_simultaneous_game(game=goofspiel(imp_info="
                  "True,num_cards=" + num_cards + ",points_order=descending))");
    }
    if (flag.substr(0, 10) == "liars_dice" && flag.length() > 12) {
      auto left_bracket = flag.find('(');
      auto right_bracket = flag.find(')');
      if (left_bracket == std::string::npos || right_bracket == std::string::npos || right_bracket <= left_bracket + 1) {
        throw std::runtime_error("Invalid flag format");
      }
      std::string parameters = flag.substr(left_bracket + 1, right_bracket - left_bracket - 1);
      auto comma = parameters.find(',');

      if (comma == std::string::npos) {
          // liars_dice(S)
          return LoadGame("liars_dice(dice_sides=" + parameters + ")");
      } else {
          // liars_dice(S,D)
          std::string dice_sides = parameters.substr(0, comma);
          std::string numdice = parameters.substr(comma + 1);
          return LoadGame("liars_dice(dice_sides=" + dice_sides + ",numdice=" + numdice + ")");
      }
    }
    if(flag == "leduc_poker") {
      return LoadGame("leduc_poker");
    } 
    throw std::runtime_error("Invalid game flag: " + flag + " not in " + absl::StrJoin(implemented_games, ", "));
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
  std::shared_ptr<const open_spiel::Game> game = open_spiel::papers_with_code::GetGameFromFlag(absl::GetFlag(FLAGS_game));

  if(run_mccfr) {
    std::cout << "Running MCCFR with " << iterations << " iterations" << " and game " << game->ToString() << std::endl;
    std::cout << std::flush;
    open_spiel::papers_with_code::RunMCCFR(game, iterations);
  }
  if(run_vanilla_cfr) {
    std::cout << "Running Vanilla CFR with " << iterations << " iterations" << " and game " << game->ToString() << std::endl;
    std::cout << std::flush;
    open_spiel::papers_with_code::RunVanillaCFR(game, iterations, save_states);
  }
  if(run_rnr) {
    std::cout << "Running RNR with " << iterations << " iterations and target player " << target_player << " and game " << game->ToString() << std::endl;
    std::cout << std::flush;
    std::vector<SweepResult> results = open_spiel::papers_with_code::RunSweep(game, iterations, target_player, open_spiel::papers_with_code::RunRNR);
    open_spiel::papers_with_code::SaveSweepResults(results, "RNR", RESULTS_DIR + "rnr_results.csv");
  }
  if(run_secfr) {
    std::cout << "Running SECFRSolver with " << iterations << " iterations and target player " << target_player << " and game " << game->ToString() << std::endl;
    std::cout << std::flush;
    std::vector<SweepResult> results = open_spiel::papers_with_code::RunSweep(game, iterations, target_player, open_spiel::papers_with_code::RunSECFRSolver);
    open_spiel::papers_with_code::SaveSweepResults(results, "SECFR", RESULTS_DIR + "secfr_results.csv");
  }

  return 0;
}