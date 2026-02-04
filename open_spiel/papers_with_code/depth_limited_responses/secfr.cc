#include "secfr.h"
#include "spiel.h"
#include "spiel_utils.h"

namespace open_spiel {
namespace papers_with_code {

void SECFRSolver::EvaluateAndUpdatePolicy() {
    ++iteration_;
    SPIEL_CHECK_TRUE(alternating_updates_);

    // SE-CFR Algorithm (Algorithm 1 from the paper, simplified without HOR)
    // Outer loop: flag ∈ [i, -i] - which component we're computing
    //   - use_exploitation=true (flag=i): exploitation component using fusion
    //   - use_exploitation=false (flag=-i): safety component (standard CFR)
    // Inner loop: D ∈ [1, 2] - which player's regrets to update

    for (bool use_exploitation : {true, false}) {
        for (int updating_player : {0, 1}) {
            if (save_states_) {
                LagrRecur(*cfr_root_state_, updating_player, use_exploitation,
                         root_reach_probs_, 1.0);
            } else {
                LagrRecur(*root_state_, updating_player, use_exploitation,
                         root_reach_probs_, 1.0);
            }
            if (regret_matching_plus_) {
                ApplyRegretMatchingPlusReset();
                ApplyRegretMatchingPlusResetOpponentInfoStates();
            }
            ApplyRegretMatching();
            ApplyRegretMatchingOpponentInfostates();
        }
    }
}

template <typename StateType>
double SECFRSolver::LagrRecur(
    StateType& state, int updating_player, bool use_exploitation,
    const std::vector<double>& reach_probabilities,
    double fixed_opponent_reach) {

    if (state.IsTerminal()) {
        return state.Returns()[updating_player];
    }

    if (state.IsChanceNode()) {
        ActionsAndProbs actions_and_probs = state.ChanceOutcomes();
        double expected_return = 0.0;
        for (const std::pair<Action, double>& action_and_prob : actions_and_probs) {
            std::vector<double> new_reach_probabilities = reach_probabilities;
            new_reach_probabilities[chance_player_] *= action_and_prob.second;
            expected_return += action_and_prob.second * LagrRecur(*state.Child(action_and_prob.first), updating_player, use_exploitation,
                            new_reach_probabilities, fixed_opponent_reach);
        }
        return expected_return;
    }

    int current_player = state.CurrentPlayer();
    int opponent_of_updating = 1 - updating_player;
    std::string info_state = state.InformationStateString(current_player);
    std::vector<Action> legal_actions = state.LegalActions(current_player);

    // Get current iteration's policy
    std::vector<double> current_policy = GetPolicy(info_state, legal_actions);
    std::vector<double> opponent_policy = GetOpponentPolicy(info_state, legal_actions);

    // Get fixed opponent policy (only relevant when current_player is opponent of updating_player)
    std::vector<double> fixed_policy = GetFixedPlayerPolicy(info_state, legal_actions);

    double expected_return = 0.0;
    std::vector<double> child_values(legal_actions.size(), 0.0);

    // Compute fusion reach at this info state (before action)
    // fusion_reach = (1-λ)π^{σ^t}_{-i}(I) + λπ^{σ̃}_{-i}(I)
    // This is only meaningful when current_player == opponent_of_updating
    double current_opponent_reach = reach_probabilities[opponent_of_updating];
    double fusion_reach_at_I = (1.0 - p_) * current_opponent_reach + p_ * fixed_opponent_reach;

    for (int action_idx = 0; action_idx < legal_actions.size(); action_idx++) {
        Action action = legal_actions[action_idx];
        std::vector<double> new_reach_probabilities = reach_probabilities;
        double new_fixed_opponent_reach = fixed_opponent_reach;

        // Update reach probabilities based on who is acting
        if(use_exploitation) {
            if(current_player == updating_player) {
                new_reach_probabilities[current_player] *= current_policy[action_idx];
            } else {
                new_reach_probabilities[current_player] *= opponent_policy[action_idx];
            }
        } else {
            if(current_player == updating_player) {
                new_reach_probabilities[current_player] *= opponent_policy[action_idx];
            } else {
                new_reach_probabilities[current_player] *= current_policy[action_idx];
            }
        }

        // Update fixed opponent reach only when current player is the opponent
        if (current_player == opponent_of_updating) {
            new_fixed_opponent_reach *= fixed_policy[action_idx];
        }

        child_values[action_idx] = LagrRecur(*state.Child(action), updating_player, use_exploitation,
                                            new_reach_probabilities, new_fixed_opponent_reach);

        // Compute expected return based on Algorithm 2 line 15
        if(!use_exploitation) {
            // F = -i (safety mode): use the non exploitation policy
            if(current_player == updating_player) {
                expected_return += opponent_policy[action_idx] * child_values[action_idx];
            } else {
                expected_return += current_policy[action_idx] * child_values[action_idx];
            }
        } else {
            if (current_player == opponent_of_updating) {
                // F = i (exploitation mode) and at opponent's node: use fusion strategy
                // v(I) += v(I,a) * π^fus_{-i}(I·a) / π^fus_{-i}(I)
                double new_opponent_reach = new_reach_probabilities[opponent_of_updating];
                double fusion_reach_at_Ia = (1.0 - p_) * new_opponent_reach + p_ * new_fixed_opponent_reach;

                if (fusion_reach_at_I > 0) {
                    expected_return += child_values[action_idx] * fusion_reach_at_Ia / fusion_reach_at_I;
                }
            } else {
                // at updating player's node: use current policy
                expected_return += current_policy[action_idx] * child_values[action_idx];
            }
        }
    }

    // Update regrets and average strategy only at updating_player's nodes (line 17: ρ(I) == D)
    if (current_player == updating_player) {
        algorithms::CFRInfoStateValues is_vals;
        if(use_exploitation) {
            is_vals = info_states_[info_state];
        } else {
            is_vals = opponent_info_states_[info_state];
        }
        
        SPIEL_CHECK_FALSE(is_vals.empty());

        const double self_reach_prob = reach_probabilities[current_player];
        double cfr_reach_prob = reach_probabilities[chance_player_];

        // Compute opponent's reach for CFR weighting (Algorithm 2 lines 19-20)
        if (use_exploitation) {
            // F = i: use fusion reach
            // σ^fus_{-i} = (1-λ)π_{opponent} + λπ^{fixed}_{opponent}
            // π_{-i} = π_c * σ^fus_{-i}
            double fusion_opponent_reach = (1.0 - p_) * reach_probabilities[opponent_of_updating]
                                          + p_ * fixed_opponent_reach;
            cfr_reach_prob *= fusion_opponent_reach;
        } else {
            // F = -i: standard CFR, use regular opponent reach
            cfr_reach_prob *= reach_probabilities[opponent_of_updating];
        }

        for (int action_idx = 0; action_idx < legal_actions.size(); action_idx++) {
            // Update regrets (Algorithm 2 line 23)
            double cfr_regret = cfr_reach_prob * (child_values[action_idx] - expected_return);
            is_vals.cumulative_regrets[action_idx] += cfr_regret;

            // Update average policy (Algorithm 2 line 22)
            if (linear_averaging_) {
                is_vals.cumulative_policy[action_idx] +=
                    iteration_ * self_reach_prob * current_policy[action_idx];
            } else {
                is_vals.cumulative_policy[action_idx] +=
                    self_reach_prob * current_policy[action_idx];
            }
        }
        if(use_exploitation) {
            info_states_[info_state] = is_vals;
        } else {
            opponent_info_states_[info_state] = is_vals;
        }
        
    }

    return expected_return;
}

std::vector<double> SECFRSolver::GetFixedPlayerPolicy(std::string info_state, std::vector<Action> legal_actions) {
    std::vector<double> fixed_player_policy(legal_actions.size(), 0.0);
    ActionsAndProbs policy = fixed_opponent_policy_->GetStatePolicy(info_state);
    for (int action_idx = 0; action_idx < legal_actions.size(); action_idx++) {
        Action action = legal_actions[action_idx];
        // Handle case where policy might not have exact action ordering
        for (const auto& ap : policy) {
            if (ap.first == action) {
                fixed_player_policy[action_idx] = ap.second;
                break;
            }
        }
    }
    return fixed_player_policy;
}

std::vector<double> SECFRSolver::GetOpponentPolicy(
    const std::string& info_state, const std::vector<Action>& legal_actions) {
  auto entry = opponent_info_states_.find(info_state);
  if (entry == opponent_info_states_.end()) {
    opponent_info_states_[info_state] = algorithms::CFRInfoStateValues(legal_actions);
    entry = opponent_info_states_.find(info_state);
  }

  SPIEL_CHECK_FALSE(entry == opponent_info_states_.end());
  SPIEL_CHECK_FALSE(entry->second.empty());
  SPIEL_CHECK_FALSE(entry->second.current_policy.empty());
  return entry->second.current_policy;
}

void SECFRSolver::ApplyRegretMatchingPlusResetOpponentInfoStates() {
    for (auto& entry : opponent_info_states_) {
      for (int aidx = 0; aidx < entry.second.num_actions(); ++aidx) {
        if (entry.second.cumulative_regrets[aidx] < 0) {
          entry.second.cumulative_regrets[aidx] = 0;
        }
      }
    }
  }
  
  void SECFRSolver::ApplyRegretMatchingOpponentInfostates() {
    for (auto& entry : opponent_info_states_) {
      entry.second.ApplyRegretMatching();
    }
  }

// Explicit template instantiations
template double SECFRSolver::LagrRecur<State>(
    State& state, int updating_player, bool use_exploitation,
    const std::vector<double>& reach_probabilities,
    double fixed_opponent_reach);

template double SECFRSolver::LagrRecur<algorithms::CfrState>(
    algorithms::CfrState& state, int updating_player, bool use_exploitation,
    const std::vector<double>& reach_probabilities,
    double fixed_opponent_reach);

}  // namespace papers_with_code
}  // namespace open_spiel
