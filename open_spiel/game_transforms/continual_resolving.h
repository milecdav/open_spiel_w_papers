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

#ifndef OPEN_SPIEL_GAME_TRANSFORMS_CONTINUAL_RESOLVING_H_
#define OPEN_SPIEL_GAME_TRANSFORMS_CONTINUAL_RESOLVING_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "open_spiel/game_transforms/matrix_valued_states.h"
#include "open_spiel/game_transforms/subgame_utils.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"

// Continual Depth-Limited Resolving (CDBR & CDRNR)
//
// Based on: "Continual Depth-limited Responses for Computing
// Counter-strategies in Sequential Games" (Milec, Kubicek, Lisy 2024).
//
// Two algorithms:
//   CDBR: Iteratively solve subgames with the opponent fixed to a model.
//         Equivalent to solver=kRNR, gadget=kNone, p=1.0.
//   CDRNR: Same structure but using RNR to balance exploitation (prob p
//          of fixed opponent) vs safety (prob 1-p of best-responding
//          opponent). Uses a gadget at subgame boundaries.
//
// The main entry point is ContinualResolve(), which:
//   1. Solves the trunk using MVS (depth-limited game)
//   2. Decomposes at the depth boundary
//   3. Re-solves each subgame using ResolveSubgames() with the chosen
//      solver (CFR, RNR) and gadget type (none, resolving, max-margin)

namespace open_spiel {

// ============================================================================
// GadgetPolicyWrapper: Adapts a policy to work with gadget info states
// ============================================================================

// When running RNR on a gadget game, the fixed opponent policy needs to
// handle gadget-prefixed info states. This wrapper:
//   - Strips the subgame prefix (e.g. "gadget_F:subgame:") and delegates
//     to the underlying policy
//   - At T/F choice nodes: always returns Follow with probability 1.0
//   - At max-margin info set choice: returns uniform over info sets
//   - At other gadget-specific nodes: falls back to uniform
//
// The underlying policy must support GetStatePolicy(string) for all
// relevant info states (e.g. a TabularPolicy from CFR).
class GadgetPolicyWrapper : public Policy {
 public:
  // subgame_prefix: prefix for subgame info states
  //   "gadget_F:subgame:" for resolving gadget
  //   "mm_F:subgame:" for max-margin gadget
  //   "unsafe:subgame:" for unsafe subgame
  // follow_action: the Follow action index for resolving gadget (1),
  //   or -1 if no T/F choice exists (max-margin, unsafe)
  // num_gadget_choice_actions: number of info-set choice actions for
  //   max-margin gadget, or 0 if not applicable
  GadgetPolicyWrapper(const Policy* underlying,
                      std::string subgame_prefix,
                      int follow_action = -1,
                      int num_gadget_choice_actions = 0);

  ActionsAndProbs GetStatePolicy(const std::string& info_state) const override;
  ActionsAndProbs GetStatePolicy(const State& state,
                                 Player player) const override;

 private:
  const Policy* underlying_;
  std::string subgame_prefix_;
  int follow_action_;
  int num_gadget_choice_actions_;
};

// ============================================================================
// ResolvingConfig: Unified configuration for subgame resolving
// ============================================================================

enum class SolverType { kCFR, kRNR };
enum class GadgetType { kNone, kResolving, kMaxMargin };

struct ResolvingConfig {
  SolverType solver = SolverType::kCFR;
  GadgetType gadget = GadgetType::kResolving;
  const Policy* opponent_model = nullptr;  // Required for kRNR
  double p = 0.5;                          // RNR restriction parameter
  int target_player = 0;                   // Who we optimize for in RNR
  int cfr_iterations = 500;
};

// ============================================================================
// ResolveSubgames: Unified subgame resolution
// ============================================================================

// Re-solve all subgames in a decomposition using the specified solver and
// gadget type. Copies trunk policy, then builds gadget games per subgame
// group, runs the solver, and extracts strategies.
//
// For kCFR: standard equilibrium-finding CFR on the gadget (both players)
// For kRNR: RNR with GadgetPolicyWrapper(opponent_model) as fixed opponent
//           (only target_player's strategy is computed; opponent uses model)
//
// Returns a combined policy covering both trunk and all subgames.
std::shared_ptr<TabularPolicy> ResolveSubgames(
    const SubgameDecomposition& decomp,
    const TabularPolicy& trunk_policy,
    const ResolvingConfig& config);

// ============================================================================
// ContinualResolve: The main continual resolving loop
// ============================================================================

// Single-level continual resolving: solve trunk with MVS, then resolve
// subgames at the depth boundary.
//
// When opponent_model is set in config, the opponent's trunk strategy is
// fixed to the model using SetFixedPolicy. The target player's trunk
// strategy is then optimized (best response in trunk). Subgames are
// resolved using ResolveSubgames with the specified solver/gadget.
//
// For CDBR: solver=kRNR, gadget=kNone, p=1.0
//   → trunk: target's best response, subgames: best response to model
// For CDRNR: solver=kRNR, gadget=kResolving or kMaxMargin, 0<p<1
//   → trunk: target's best response, subgames: RNR tradeoff with gadget
//
// Parameters:
//   game: The original game
//   opponent_model: The opponent's suspected strategy
//   config: Solver configuration (solver type, gadget type, p, etc.)
//   depth: Depth at which to decompose
//   depth_mode: How to measure depth (action-based or round-based)
std::shared_ptr<TabularPolicy> ContinualResolve(
    std::shared_ptr<const Game> game,
    const Policy& opponent_model,
    const ResolvingConfig& config,
    int depth,
    MVSGame::DepthMode depth_mode);

}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAME_TRANSFORMS_CONTINUAL_RESOLVING_H_
