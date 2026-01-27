#include "secfr.h"
#include "spiel.h"
#include "spiel_utils.h"

namespace open_spiel {
namespace papers_with_code {

void SECFRSolver::EvaluateAndUpdatePolicy() {
    ++iteration_;
    SPIEL_CHECK_TRUE(alternating_updates_);
    for (bool updating_current_player : {true, false}) {
        for(int updating_player : {0, 1}) {
            std::vector<double> new_reach_probabilities = root_reach_probs_;
            if(!updating_current_player) {
                new_reach_probabilities[fixed_player_index_] = 0.0;
            }
            if(save_states_) {
                LagrRecur(*cfr_root_state_, updating_player, updating_current_player, new_reach_probabilities);
            } else {
                LagrRecur(*root_state_, updating_player, updating_current_player, new_reach_probabilities);
            }
            if (regret_matching_plus_) {
                ApplyRegretMatchingPlusReset();
            }
            ApplyRegretMatching();
        }
    }
}

template <typename StateType>
double SECFRSolver::LagrRecur(
    StateType& state, int updating_player, bool updating_current_player,
    const std::vector<double>& reach_probabilities) {
    if(state.IsTerminal()) {
        return state.Returns()[updating_player];
    }
    if(state.IsChanceNode()) {
        ActionsAndProbs actions_and_probs = state.ChanceOutcomes();
        std::pair<Action, double> action_and_prob = open_spiel::SampleAction(actions_and_probs, rng_);
        std::vector<double> new_reach_probabilities = reach_probabilities;
        new_reach_probabilities[chance_player_] *= action_and_prob.second;
        return LagrRecur(*state.Child(action_and_prob.first), updating_player, updating_current_player, new_reach_probabilities);
    }
    int current_player = state.CurrentPlayer();
    std::string info_state = state.InformationStateString(current_player);
    std::vector<Action> legal_actions = state.LegalActions(current_player);
    std::vector<double> infostate_policy = GetPolicy(info_state, legal_actions);
    std::vector<double> fixed_player_policy = GetFixedPlayerPolicy(info_state, legal_actions);
    reach_probabilities_cache_[info_state] = 
            p_ * reach_probabilities[fixed_player_index_] + (1 - p_) * reach_probabilities[current_player];
    double expected_return = 0.0;
    std::vector<double> child_values(legal_actions.size(), 0.0);
    for(int action_idx = 0; action_idx < legal_actions.size(); action_idx++) {
        Action action = legal_actions[action_idx];
        std::vector<double> new_reach_probabilities = reach_probabilities;
        new_reach_probabilities[current_player] *= infostate_policy[action_idx];
        new_reach_probabilities[fixed_player_index_] *= fixed_player_policy[action_idx];
        reach_probabilities_cache_[info_state + "|" + std::to_string(action)] = 
            p_ * new_reach_probabilities[fixed_player_index_] + (1 - p_) * new_reach_probabilities[current_player];
        
        child_values[action_idx] = LagrRecur(*state.Child(action), updating_player, updating_current_player, new_reach_probabilities);
        if(updating_current_player && current_player != updating_player) {
            expected_return += reach_probabilities_cache_[info_state + "|" + std::to_string(action)] / 
                reach_probabilities_cache_[info_state] * child_values[action_idx];
        } else {
            expected_return += infostate_policy[action_idx] * child_values[action_idx];
        }
    }

    if(current_player == updating_player) {
        algorithms::CFRInfoStateValues is_vals = info_states_[info_state];
        SPIEL_CHECK_FALSE(is_vals.empty());
        const double self_reach_prob = reach_probabilities[current_player];
        double cfr_reach_prob = reach_probabilities[chance_player_];
        if(updating_current_player) {
            // Safety component: weighted by (1-p)
            cfr_reach_prob *= (1 - p_) * reach_probabilities[1 - current_player] + p_ * reach_probabilities[fixed_player_index_];
        } else {
            cfr_reach_prob *= reach_probabilities[1 - current_player];
        }
        for(int action_idx = 0; action_idx < legal_actions.size(); action_idx++) {
            double cfr_regret = cfr_reach_prob *
                (child_values[action_idx] - expected_return);
            is_vals.cumulative_regrets[action_idx] += cfr_regret;
            // Update average policy.
            if (linear_averaging_) {
                is_vals.cumulative_policy[action_idx] +=
                    iteration_ * self_reach_prob * infostate_policy[action_idx];
            } else {
                is_vals.cumulative_policy[action_idx] +=
                    self_reach_prob * infostate_policy[action_idx];
            }
        }
        info_states_[info_state] = is_vals;
    }
    return expected_return;    
}


std::vector<double> SECFRSolver::GetFixedPlayerPolicy(std::string info_state, std::vector<Action> legal_actions) {
    std::vector<double> fixed_player_policy(legal_actions.size(), 0.0);
    ActionsAndProbs policy = fixed_opponent_policy_->GetStatePolicy(info_state);
    for(int action_idx = 0; action_idx < legal_actions.size(); action_idx++) {
        Action action = legal_actions[action_idx];
        SPIEL_CHECK_EQ(action, policy[action_idx].first);
        fixed_player_policy[action_idx] = policy[action_idx].second;
    }
    return fixed_player_policy;
}

// Explicit template instantiations
template double SECFRSolver::LagrRecur<State>(
    State& state, int updating_player, bool updating_current_player,
    const std::vector<double>& reach_probabilities);

template double SECFRSolver::LagrRecur<algorithms::CfrState>(
    algorithms::CfrState& state, int updating_player, bool updating_current_player,
    const std::vector<double>& reach_probabilities);

}  // namespace papers_with_code
}  // namespace open_spiel