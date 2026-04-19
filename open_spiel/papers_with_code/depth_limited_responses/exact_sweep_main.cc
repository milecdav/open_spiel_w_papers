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

// exact_sweep_main.cc
// Sweeps p in {0,0.1,...,1} for SES, CDRNR, OX-Search using ContinualResolve
// (LP-based exact solving). Outputs a CSV with gain and exploitability.

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"

ABSL_FLAG(std::string, game, "kuhn_poker", "Game to run");
ABSL_FLAG(int, depth, 2, "Depth at which to decompose");
ABSL_FLAG(std::string, depth_mode, "action",
          "Depth mode: 'action' or 'round'");
ABSL_FLAG(int, target_player, 0, "Target player (0 or 1)");
ABSL_FLAG(std::string, out, "results/exact_sweep.csv", "Output CSV path");
ABSL_FLAG(std::string, algorithms, "ses,cdrnr,ox",
          "Comma-separated list of algorithms to run: ses,cdrnr,ox");
ABSL_FLAG(std::string, p_values,
          "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
          "Comma-separated p values to sweep");
ABSL_FLAG(bool, smoke, false,
          "Smoke test: only run p in {0.0, 0.5, 1.0} per algorithm");

#include "algorithms/best_response.h"
#include "algorithms/expected_returns.h"
#include "algorithms/tabular_exploitability.h"
#if OPEN_SPIEL_BUILD_WITH_ORTOOLS
#include "algorithms/ortools/sequence_form_lp.h"
#endif
#include "game_transforms/continual_resolving.h"
#include "game_transforms/matrix_valued_states.h"
#include "game_transforms/turn_based_simultaneous_game.h"
#include "policy.h"
#include "spiel.h"
#include "spiel_utils.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace open_spiel {
namespace exact_sweep {
namespace {

// A policy that falls back to uniform when the primary policy has no entry.
// Uses the State-based overload so it can enumerate legal actions for fallback.
class UniformFallbackPolicy : public Policy {
 public:
  explicit UniformFallbackPolicy(const Policy* primary) : primary_(primary) {}

  ActionsAndProbs GetStatePolicy(const State& state,
                                 Player player) const override {
    auto result = primary_->GetStatePolicy(state, player);
    if (!result.empty()) return result;
    // Fallback: uniform over legal actions for this player.
    std::vector<Action> legal = state.LegalActions(player);
    if (legal.empty()) return {};
    ActionsAndProbs uniform;
    uniform.reserve(legal.size());
    double prob = 1.0 / legal.size();
    for (Action a : legal) uniform.push_back({a, prob});
    return uniform;
  }

  ActionsAndProbs GetStatePolicy(const std::string& info_state) const override {
    return primary_->GetStatePolicy(info_state);
  }

 private:
  const Policy* primary_;
};

// A policy that uses target_policy for target_player and uniform for opponent.
class MixedPolicy : public Policy {
 public:
  MixedPolicy(const Policy* target_policy, int target_player,
              const Policy* uniform_policy)
      : target_policy_(target_policy),
        target_player_(target_player),
        uniform_policy_(uniform_policy) {}

  ActionsAndProbs GetStatePolicy(const State& state,
                                 Player player) const override {
    if (player == target_player_) {
      return target_policy_->GetStatePolicy(state, player);
    }
    return uniform_policy_->GetStatePolicy(state, player);
  }

  ActionsAndProbs GetStatePolicy(const std::string& info_state) const override {
    // Fallback: try target policy first, then uniform
    auto result = target_policy_->GetStatePolicy(info_state);
    if (!result.empty()) return result;
    return uniform_policy_->GetStatePolicy(info_state);
  }

 private:
  const Policy* target_policy_;
  int target_player_;
  const Policy* uniform_policy_;
};

struct SweepRow {
  std::string game;
  std::string algorithm;
  double p;
  double gain;
  double exploitability;  // BR_opp_value - game_value_opp (gain over Nash)
  double br_opp_value;    // raw BR value for opponent
  int target_player;
  int depth;
  std::string depth_mode;
};

ResolvingConfig MakeSESConfig(int target_player, const Policy* uniform,
                              double p) {
  ResolvingConfig cfg;
  cfg.gadget_kind = GadgetKind::kMaxMargin;
  cfg.response_kind = ResponseKind::kRNR;
  cfg.solver_kind = SolverKind::kLP;
  cfg.lock_opponent_in_fixed_branch = false;
  cfg.target_player = target_player;
  cfg.opponent_model = uniform;
  cfg.p = p;
  // Legacy transitional fields
  cfg.solver = SolverType::kLP;
  cfg.gadget = GadgetType::kMaxMargin;
  return cfg;
}

ResolvingConfig MakeCDRNRConfig(int target_player, const Policy* uniform,
                                double p) {
  ResolvingConfig cfg;
  cfg.gadget_kind = GadgetKind::kResolving;
  cfg.response_kind = ResponseKind::kRNR;
  cfg.solver_kind = SolverKind::kLP;
  cfg.lock_opponent_in_fixed_branch = true;
  cfg.target_player = target_player;
  cfg.opponent_model = uniform;
  cfg.p = p;
  // Legacy transitional fields
  cfg.solver = SolverType::kLP;
  cfg.gadget = GadgetType::kResolving;
  return cfg;
}

ResolvingConfig MakeOXConfig(int target_player, const Policy* uniform,
                             double p) {
  ResolvingConfig cfg;
  cfg.gadget_kind = GadgetKind::kResolvingByIS;
  cfg.response_kind = ResponseKind::kRNR;
  cfg.solver_kind = SolverKind::kLP;
  cfg.lock_opponent_in_fixed_branch = false;
  cfg.target_player = target_player;
  cfg.opponent_model = uniform;
  cfg.p = p;
  // Legacy transitional fields
  cfg.solver = SolverType::kLP;
  cfg.gadget = GadgetType::kResolvingByIS;
  return cfg;
}

void WriteRow(std::ofstream& file, const SweepRow& row) {
  std::string line = row.game + "," + row.algorithm + "," +
                     std::to_string(row.p) + "," + std::to_string(row.gain) +
                     "," + std::to_string(row.exploitability) + "," +
                     std::to_string(row.br_opp_value) + "," +
                     std::to_string(row.target_player) + "," +
                     std::to_string(row.depth) + "," + row.depth_mode + "\n";
  file << line;
  file.flush();
  std::cout << line;
  std::cout.flush();
}

}  // namespace
}  // namespace exact_sweep
}  // namespace open_spiel

int main(int argc, char** argv) {
  // Install a throwing error handler so SpielFatalError becomes catchable.
  open_spiel::SetErrorHandler([](const std::string& msg) {
    throw std::runtime_error(msg);
  });

  std::vector<char*> positional_args = absl::ParseCommandLine(argc, argv);

  std::string game_str = absl::GetFlag(FLAGS_game);
  int depth = absl::GetFlag(FLAGS_depth);
  std::string depth_mode_str = absl::GetFlag(FLAGS_depth_mode);
  int target_player = absl::GetFlag(FLAGS_target_player);
  std::string out_path = absl::GetFlag(FLAGS_out);
  std::string algs_str = absl::GetFlag(FLAGS_algorithms);
  std::string p_values_str = absl::GetFlag(FLAGS_p_values);
  bool smoke = absl::GetFlag(FLAGS_smoke);

  // Parse depth_mode
  open_spiel::MVSGame::DepthMode depth_mode;
  if (depth_mode_str == "action") {
    depth_mode = open_spiel::MVSGame::DepthMode::kActionBased;
  } else if (depth_mode_str == "round") {
    depth_mode = open_spiel::MVSGame::DepthMode::kRoundBased;
  } else {
    std::cerr << "Unknown depth_mode: " << depth_mode_str
              << ". Use 'action' or 'round'.\n";
    return 1;
  }

  // Parse algorithms
  std::vector<std::string> algorithms =
      absl::StrSplit(algs_str, ',', absl::SkipEmpty());

  // Parse p_values
  std::vector<double> p_values;
  if (smoke) {
    p_values = {0.0, 0.5, 1.0};
  } else {
    for (const auto& s : absl::StrSplit(p_values_str, ',', absl::SkipEmpty())) {
      p_values.push_back(std::stod(std::string(s)));
    }
  }

  // Load game. Auto-convert simultaneous-move games to turn-based: the
  // InfostateTree / gadget / MVS / RNRMixture stack is built on the assumption
  // of sequential play.
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame(game_str);
  if (game->GetType().dynamics ==
      open_spiel::GameType::Dynamics::kSimultaneous) {
    std::cout << "Converting simultaneous-move game to turn-based.\n";
    game = open_spiel::ConvertToTurnBased(*game);
  }
  std::cout << "Game: " << game->ToString() << "\n";
  std::cout << "Depth: " << depth << ", Mode: " << depth_mode_str << "\n";
  std::cout << "Target player: " << target_player << "\n";
  std::cout << "Algorithms: " << algs_str << "\n";
  std::cout << "Smoke: " << (smoke ? "true" : "false") << "\n\n";

  // Build uniform policy
  auto uniform = std::make_shared<open_spiel::UniformPolicy>();

  // Ensure output directory exists
  {
    std::filesystem::path out_dir =
        std::filesystem::path(out_path).parent_path();
    if (!out_dir.empty()) {
      std::error_code ec;
      std::filesystem::create_directories(out_dir, ec);
      // Ignore ec: directory may already exist or be a special path like /tmp
    }
  }

  // Open output file
  std::ofstream out_file(out_path);
  if (!out_file.is_open()) {
    std::cerr << "Failed to open output file: " << out_path << "\n";
    return 1;
  }

  // Write CSV header
  out_file << "game,algorithm,p,gain,exploitability,br_opp_value,target_player,depth,depth_mode\n";
  out_file.flush();
  std::cout << "game,algorithm,p,gain,exploitability,br_opp_value,target_player,depth,depth_mode\n";
  std::cout.flush();

  // Compute opponent Nash value once (same for all algorithms and p values).
  int opponent = 1 - target_player;
  double game_value_opp = 0.0;
#if OPEN_SPIEL_BUILD_WITH_ORTOOLS
  {
    open_spiel::algorithms::ortools::SequenceFormLpSpecification spec(*game);
    spec.SpecifyLinearProgram(opponent);
    game_value_opp = spec.Solve();
    std::cout << "Nash value for opponent (player " << opponent
              << "): " << game_value_opp << "\n\n";
  }
#else
  std::cout << "WARNING: OR-Tools not available; game_value_opp=0 (exploitability will equal br_opp_value).\n\n";
#endif

  for (const std::string& alg : algorithms) {
    std::cout << "\n=== Algorithm: " << alg << " ===\n";
    std::cout.flush();

    for (double p : p_values) {
      std::cout << "  p=" << p << " ... ";
      std::cout.flush();

      // Build config
      open_spiel::ResolvingConfig config;
      if (alg == "ses") {
        config = open_spiel::exact_sweep::MakeSESConfig(target_player,
                                                         uniform.get(), p);
      } else if (alg == "cdrnr") {
        config = open_spiel::exact_sweep::MakeCDRNRConfig(target_player,
                                                           uniform.get(), p);
      } else if (alg == "ox") {
        config = open_spiel::exact_sweep::MakeOXConfig(target_player,
                                                        uniform.get(), p);
      } else {
        std::cerr << "Unknown algorithm: " << alg << "\n";
        continue;
      }

      // Run ContinualResolve
      std::shared_ptr<open_spiel::TabularPolicy> target_policy;
      try {
        target_policy = open_spiel::ContinualResolve(
            game, *uniform, config, depth, depth_mode);
      } catch (const std::exception& e) {
        std::cerr << "ContinualResolve failed for alg=" << alg << " p=" << p
                  << ": " << e.what() << "\n";
        continue;
      }

      // Wrap target_policy with uniform fallback so missing info states
      // (unreachable under the resolved joint reach) don't crash.
      open_spiel::exact_sweep::UniformFallbackPolicy target_with_fallback(
          target_policy.get());

      // Compute gain: E[u_target | target plays target_policy, opp plays uniform]
      std::vector<const open_spiel::Policy*> policies_for_gain;
      if (target_player == 0) {
        policies_for_gain = {&target_with_fallback, uniform.get()};
      } else {
        policies_for_gain = {uniform.get(), &target_with_fallback};
      }

      double gain = open_spiel::algorithms::ExpectedReturns(
          *game->NewInitialState(), policies_for_gain,
          /*depth_limit=*/-1,
          /*use_infostate_get_policy=*/false)[target_player];

      // Compute BR value for opponent when target plays target_policy.
      // Use TabularBestResponse with the fallback wrapper's tabular expansion.
      // TabularBestResponse uses GetStatePolicy(State&) internally.
      open_spiel::algorithms::TabularBestResponse br(
          *game, opponent, &target_with_fallback);
      double br_opp_value = br.Value(*game->NewInitialState());
      // Exploitability = gain over Nash for the opponent.
      double exploitability = br_opp_value - game_value_opp;

      open_spiel::exact_sweep::SweepRow row;
      row.game = game_str;
      row.algorithm = alg;
      row.p = p;
      row.gain = gain;
      row.exploitability = exploitability;
      row.br_opp_value = br_opp_value;
      row.target_player = target_player;
      row.depth = depth;
      row.depth_mode = depth_mode_str;

      open_spiel::exact_sweep::WriteRow(out_file, row);
    }
  }

  out_file.close();
  std::cout << "\nDone. Results written to " << out_path << "\n";
  return 0;
}
