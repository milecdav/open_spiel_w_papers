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

#ifndef OPEN_SPIEL_GAME_TRANSFORMS_GADGET_H_
#define OPEN_SPIEL_GAME_TRANSFORMS_GADGET_H_

// Abstract Gadget interface and concrete strategy implementations.
// This file is a skeleton — full implementation will be added in later steps.

#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "open_spiel/game_transforms/full_gadget.h"
#include "open_spiel/game_transforms/subgame_utils.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"

namespace open_spiel {

namespace algorithms {
class TabularBestResponse;
}  // namespace algorithms

// Forward declaration for ResolvingConfig (defined in continual_resolving.h)
struct ResolvingConfig;

// Per-group state used to build a gadget.
struct GadgetContext {
  std::shared_ptr<const Game> original_game;
  // Roots for this group (owned by caller, NOT by ctx).
  const std::vector<std::unique_ptr<State>>* roots = nullptr;
  std::string pub_obs;

  // From the trunk decomposition.
  std::array<const std::unordered_map<std::string, double>*, 2> cfvs{nullptr, nullptr};
  std::array<const std::unordered_map<std::string, double>*, 2> reach_probs{nullptr, nullptr};
  const std::unordered_map<std::string, double>* chance_reach = nullptr;

  // Which player we're extracting the strategy for in this build.
  Player resolving_player = 0;

  // Optional. Present only when the caller says so.
  const Policy* opponent_model = nullptr;
  const algorithms::TabularBestResponse* br_non_res = nullptr;

  // For MaxMargin SES-reproduction mode.
  bool reach_weighted_is_pick = false;
};

// Abstract gadget interface.
class Gadget {
 public:
  virtual ~Gadget() = default;

  // Build the gadget game. Return nullptr to skip this group.
  virtual std::shared_ptr<const Game> Build(const GadgetContext& ctx) const = 0;

  // Prefix used to key subgame info states inside the gadget game.
  virtual std::string SubgamePrefix() const = 0;

  // True if the non-resolving player's strategy inside the gadget is
  // distorted (because they also use gadget-specific actions). When true,
  // ResolveSubgames must do a second CFR pass with the roles swapped to
  // extract the opponent's undistorted subgame strategy.
  virtual bool OpponentStrategyIsDistorted() const { return true; }

  virtual const char* Name() const = 0;
};

// Factory functions for concrete gadget strategies.
std::unique_ptr<Gadget> MakeUnsafeGadget();
std::unique_ptr<Gadget> MakeResolvingGadget();
std::unique_ptr<Gadget> MakeMaxMarginGadget();
std::unique_ptr<Gadget> MakeResolvingByISGadget();
std::unique_ptr<Gadget> MakeFullGadget(
    FullGadgetGame::Mode mode,
    std::shared_ptr<const Policy> trunk_policy,
    std::unordered_map<std::string, std::vector<std::string>>
        all_boundary_states_by_group);

// Factory from config (selects correct gadget and passes required extras).
// ResolvingConfig is defined in continual_resolving.h; forward declared here
// to avoid circular dependencies. Callers must include continual_resolving.h.
// std::unique_ptr<Gadget> MakeGadgetFromConfig(
//     const ResolvingConfig& config,
//     const SubgameDecomposition& decomp,
//     const TabularPolicy& trunk_policy);

}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAME_TRANSFORMS_GADGET_H_
