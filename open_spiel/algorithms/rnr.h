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
// Based on: "Computing Robust Counter-Strategies" by Johanson, Zinkevich,
// and Bowling (NIPS 2007).
// https://poker.cs.ualberta.ca/publications/NIPS07-rnash.pdf
//
// RNR computes a strategy that balances exploiting a suspected opponent
// tendency (fixed policy) while bounding worst-case performance.
// The opponent is modeled as playing:
//   - The fixed policy with probability p
//   - A free best-response with probability (1-p)

#ifndef OPEN_SPIEL_ALGORITHMS_RNR_H_
#define OPEN_SPIEL_ALGORITHMS_RNR_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace algorithms {

// Restricted Nash Response Solver
//
// Computes a p-restricted Nash response for `target_player` against a fixed
// opponent policy. The opponent is modeled as playing the fixed policy with
// probability p and best-responding with probability (1-p).
class RNRSolver : public CFRSolverBase {
 public:
  RNRSolver(const Game& game, Player target_player,
            const Policy* fixed_opponent_policy, double p, bool alternating_updates = true,
            bool linear_averaging = false, bool regret_matching_plus = false,
            bool save_states = false, bool random_initial_regrets = false, int seed = 0)
    : CFRSolverBase(game, alternating_updates, linear_averaging, regret_matching_plus, save_states, random_initial_regrets, seed),
                    target_player_(target_player),
                    opponent_player_(1 -target_player),
                    fixed_opponent_policy_(fixed_opponent_policy),                    
                    p_(p) {}

  virtual ~RNRSolver() = default;

  // Override to use RNRSolver's ComputeCounterFactualRegret
  void EvaluateAndUpdatePolicy() override;

 protected:
  Player target_player_;
  Player opponent_player_;
  const Policy* fixed_opponent_policy_;
  double p_;  // Restriction parameter

  template <typename StateType>
  std::vector<double> ComputeCounterFactualRegret(
      StateType& state, const absl::optional<int>& alternating_player,
      const std::vector<double>& reach_probabilities,
      const std::vector<const Policy*>* policy_overrides, double fixed_opponent_reach);

  // Template supports both State and CfrState.
  template <typename StateType>
  std::vector<double> ComputeCounterFactualRegretForActionProbs(
      StateType& state, const absl::optional<int>& alternating_player,
      const std::vector<double>& reach_probabilities, const int current_player,
      const std::vector<double>& info_state_policy,
      const std::vector<Action>& legal_actions,
      std::vector<double>* child_values_out,
      const std::vector<const Policy*>* policy_overrides, double fixed_opponent_reach);

  double CounterFactualReachProb(
    const std::vector<double>& reach_probabilities, const int player, const std::string& info_state);
};

}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_RNR_H_
