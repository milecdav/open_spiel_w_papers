
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

// Restricted Nash Response (RNR) algorithm implementation.
// Based on: "HORSE-CFR: Hierarchical opponent reasoning for safe exploitation counterfactual regret minimization" by Wang, Wang,
// and Song (Expert systems with applications).
// https://www.sciencedirect.com/science/article/pii/S0957417424025648
//
// SECFR computes a strategy that balances exploiting a suspected opponent
// tendency (fixed policy) while bounding worst-case performance.
// The opponent is modeled as playing:
//   - The fixed policy with probability p
//   - A free best-response with probability (1-p)

#ifndef OPEN_SPIEL_PAPERS_SECFR_H_
#define OPEN_SPIEL_PAPERS_SECFR_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace papers_with_code {

// Restricted Nash Response Solver
//
// Computes a p-restricted Nash response for `target_player` against a fixed
// opponent policy. The opponent is modeled as playing the fixed policy with
// probability p and best-responding with probability (1-p).
class SECFRSolver : public algorithms::CFRSolverBase {
 public:
  SECFRSolver(const Game& game, const Policy* fixed_opponent_policy, double p, 
    bool linear_averaging = true, bool regret_matching_plus = true, bool save_states = true, bool random_initial_regrets = false, int seed = 0) 
  : CFRSolverBase(game, true, linear_averaging, regret_matching_plus, save_states, random_initial_regrets, seed),
    fixed_opponent_policy_(fixed_opponent_policy),
    p_(p),
    fixed_player_index_(chance_player_ + 1),
    root_reach_probs_(game.NumPlayers() + 2, 1.0) {}

  virtual ~SECFRSolver() = default;

  void EvaluateAndUpdatePolicy() override;

  std::vector<double> GetFixedPlayerPolicy(std::string info_state, std::vector<Action> legal_actions);

 protected:
  const Policy* fixed_opponent_policy_;
  double p_;
  int fixed_player_index_;
  std::vector<double> root_reach_probs_;
  std::unordered_map<std::string, double> reach_probabilities_cache_;

  template <typename StateType>
  double LagrRecur(
      StateType& state, int updating_player, bool updating_current_player,
      const std::vector<double>& reach_probabilities);
};
}  // namespace papers_with_code
}  // namespace open_spiel

#endif  // OPEN_SPIEL_PAPERS_SECFR_H_
