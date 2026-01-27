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

#include <algorithm>
#include <type_traits>

#include "open_spiel/spiel_utils.h"
#include "spiel.h"

namespace open_spiel {
namespace algorithms {
namespace {
  inline constexpr double kRandomInitialRegretsMagnitude = 0.001;
}  // namespace

// Compute counterfactual regrets. Alternates recursively with
// ComputeCounterFactualRegretForActionProbs.
//
// Args:
// - state: The state to start the recursion (supports both State and CfrState).
// - alternating_player: Optionally only update this player.
// - reach_probabilities: The reach probabilities of this state for each
//      player, ending with the chance player.
//
// Returns:
//   The value of the state for each player (excluding the chance player).
template <typename StateType>
std::vector<double> RNRSolver::ComputeCounterFactualRegret(
    StateType& state, const absl::optional<int>& alternating_player,
    const std::vector<double>& reach_probabilities,
    const std::vector<const Policy*>* policy_overrides, double fixed_opponent_reach) {  
  if (state.IsTerminal()) {
    std::vector<double> returns = state.Returns();
    for (int i = 0; i < returns.size(); i++) {
      if(i == target_player_) {
        returns[i] = returns[i] * ((1.0 - p_) * reach_probabilities[1-i] + p_ * fixed_opponent_reach);
      } else {
        returns[i] = returns[i] * reach_probabilities[1-i];
      }
      returns[i] = returns[i] * reach_probabilities[chance_player_];
    }
    return returns;
  }
  if (state.IsChanceNode()) {
    ActionsAndProbs actions_and_probs = state.ChanceOutcomes();
    std::vector<double> dist(actions_and_probs.size(), 0);
    std::vector<Action> outcomes(actions_and_probs.size(), 0);
    for (int oidx = 0; oidx < actions_and_probs.size(); ++oidx) {
      outcomes[oidx] = actions_and_probs[oidx].first;
      dist[oidx] = actions_and_probs[oidx].second;
    }
    return ComputeCounterFactualRegretForActionProbs(
        state, alternating_player, reach_probabilities, chance_player_, dist,
        outcomes, nullptr, policy_overrides, fixed_opponent_reach);
  }

  int current_player = state.CurrentPlayer();
  std::string info_state = state.InformationStateString(current_player);

  if (AllPlayersHaveZeroReachProb(reach_probabilities) && (current_player != target_player_ && fixed_opponent_reach == 0.0)) {
    // The value returned is not used: if the reach probability for all players
    // is 0, then the last taken action has probability 0, so the
    // returned value is not impacting the parent node value.
    return std::vector<double>(game_->NumPlayers(), 0.0);
  }
  
  std::vector<Action> legal_actions = state.LegalActions(current_player);

  // Load current policy.
  std::vector<double> info_state_policy;
  if (policy_overrides && policy_overrides->at(current_player)) {
    GetInfoStatePolicyFromPolicy(&info_state_policy, legal_actions,
                                 policy_overrides->at(current_player),
                                 info_state);
  } else {
    info_state_policy = GetPolicy(info_state, legal_actions);
  }

  std::vector<double> child_utilities;
  child_utilities.reserve(legal_actions.size());
  const std::vector<double> state_value =
      ComputeCounterFactualRegretForActionProbs(
          state, alternating_player, reach_probabilities, current_player,
          info_state_policy, legal_actions, &child_utilities, policy_overrides, fixed_opponent_reach);

  // Perform regret and average strategy updates.
  if (!alternating_player || *alternating_player == current_player) {
    CFRInfoStateValues is_vals = info_states_[info_state];
    SPIEL_CHECK_FALSE(is_vals.empty());

    const double self_reach_prob = reach_probabilities[current_player];

    for (int aidx = 0; aidx < legal_actions.size(); ++aidx) {
      // Update regrets.
      double cfr_regret = (child_utilities[aidx] - state_value[current_player]);

      is_vals.cumulative_regrets[aidx] += cfr_regret;

      // Update average policy.
      if (linear_averaging_) {
        is_vals.cumulative_policy[aidx] +=
            iteration_ * self_reach_prob * info_state_policy[aidx];
      } else {
        is_vals.cumulative_policy[aidx] +=
            self_reach_prob * info_state_policy[aidx];
      }
    }

    info_states_[info_state] = is_vals;
  }

  return state_value;
}

// Compute counterfactual regrets given certain action probabilities.
// Alternates recursively with ComputeCounterFactualRegret.
//
// Args:
// - state: The state to start the recursion (supports both State and CfrState).
// - alternating_player: Optionally only update this player.
// - reach_probabilities: The reach probabilities of this state.
// - current_player: Either a player or chance_player_.
// - action_probs: The action probabilities to use for this state.
// - child_values_out: optional output parameter which is filled with the child
//           utilities for each action, for current_player.
// Returns:
//   The value of the state for each player (excluding the chance player).
template <typename StateType>
std::vector<double> RNRSolver::ComputeCounterFactualRegretForActionProbs(
    StateType& state, const absl::optional<int>& alternating_player,
    const std::vector<double>& reach_probabilities, const int current_player,
    const std::vector<double>& info_state_policy,
    const std::vector<Action>& legal_actions,
    std::vector<double>* child_values_out,
    const std::vector<const Policy*>* policy_overrides, double fixed_opponent_reach) {
  std::vector<double> state_value(game_->NumPlayers());
  ActionsAndProbs fixed_opponent_reach_probs;
  if(current_player == opponent_player_) {
    fixed_opponent_reach_probs = fixed_opponent_policy_->GetStatePolicy(state.InformationStateString(opponent_player_));
  }
  for (int aidx = 0; aidx < legal_actions.size(); ++aidx) {
    double new_fixed_opponent_reach = fixed_opponent_reach;
    if(current_player == opponent_player_) {
      new_fixed_opponent_reach = new_fixed_opponent_reach * fixed_opponent_reach_probs[aidx].second;
    }
    const Action action = legal_actions[aidx];
    const double prob = info_state_policy[aidx];
    std::vector<double> new_reach_probabilities(reach_probabilities);
    new_reach_probabilities[current_player] *= prob;
    auto new_state = state.Child(action);
    std::vector<double> child_value =
        ComputeCounterFactualRegret(*new_state, alternating_player,
                                    new_reach_probabilities, policy_overrides, new_fixed_opponent_reach);

    for (int i = 0; i < state_value.size(); ++i) {
      if(i == current_player) {
        state_value[i] += child_value[i] * prob;
      } else {
        state_value[i] += child_value[i];
      }
    }
    if (child_values_out != nullptr) {
      child_values_out->push_back(child_value[current_player]);
    }
  }
  return state_value;
}

void RNRSolver::EvaluateAndUpdatePolicy() {
  ++iteration_;
  if (alternating_updates_) {
    for (int player = 0; player < game_->NumPlayers(); player++) {
      if (save_states_) {
        ComputeCounterFactualRegret(*cfr_root_state_, player, root_reach_probs_,
                                    nullptr, 1.0);
      } else {
        ComputeCounterFactualRegret(*root_state_, player, root_reach_probs_,
                                    nullptr, 1.0);
      }
      if (regret_matching_plus_) {
        ApplyRegretMatchingPlusReset();
      }
      ApplyRegretMatching();
    }
  } else {
    if (save_states_) {
      ComputeCounterFactualRegret(*cfr_root_state_, absl::nullopt, root_reach_probs_,
                                  nullptr, 1.0);
    } else {
      ComputeCounterFactualRegret(*root_state_, absl::nullopt, root_reach_probs_,
                                  nullptr, 1.0);
    }
    if (regret_matching_plus_) {
      ApplyRegretMatchingPlusReset();
    }
    ApplyRegretMatching();
  }
}

}  // namespace algorithms
}  // namespace open_spiel