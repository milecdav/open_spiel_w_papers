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

#include "open_spiel/game_transforms/gadget.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/game_transforms/max_margin_gadget.h"
#include "open_spiel/game_transforms/resolving_by_is.h"
#include "open_spiel/game_transforms/resolving_gadget.h"
#include "open_spiel/game_transforms/unsafe_subgame.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {

namespace {

double ComputeJointReachFromContext(const std::string& hist,
                                    const GadgetContext& ctx) {
  SPIEL_CHECK_TRUE(ctx.reach_probs[0] != nullptr);
  SPIEL_CHECK_TRUE(ctx.reach_probs[1] != nullptr);
  double r0 = 0.0, r1 = 0.0;
  auto it0 = ctx.reach_probs[0]->find(hist);
  if (it0 != ctx.reach_probs[0]->end()) r0 = it0->second;
  auto it1 = ctx.reach_probs[1]->find(hist);
  if (it1 != ctx.reach_probs[1]->end()) r1 = it1->second;
  double joint = r0 * r1;
  if (ctx.chance_reach != nullptr) {
    auto itc = ctx.chance_reach->find(hist);
    if (itc != ctx.chance_reach->end() && itc->second > 0.0) {
      joint /= itc->second;
    }
  }
  return joint;
}

class UnsafeGadgetStrategy final : public Gadget {
 public:
  std::shared_ptr<const Game> Build(const GadgetContext& ctx) const override {
    SPIEL_CHECK_TRUE(ctx.original_game != nullptr);
    SPIEL_CHECK_TRUE(ctx.roots != nullptr);

    std::vector<std::unique_ptr<State>> subgame_roots;
    std::vector<double> total_reaches;
    for (const auto& root : *ctx.roots) {
      const std::string hist = root->HistoryString();
      double total = ComputeJointReachFromContext(hist, ctx);
      if (total > 0.0) {
        subgame_roots.push_back(root->Clone());
        total_reaches.push_back(total);
      }
    }
    if (subgame_roots.empty()) return nullptr;
    return CreateUnsafeSubgame(ctx.original_game, std::move(subgame_roots),
                               std::move(total_reaches));
  }

  std::string SubgamePrefix() const override { return "unsafe:subgame:"; }
  bool OpponentStrategyIsDistorted() const override { return false; }
  const char* Name() const override { return "Unsafe"; }
};

class ResolvingGadgetStrategy final : public Gadget {
 public:
  std::shared_ptr<const Game> Build(const GadgetContext& ctx) const override {
    SPIEL_CHECK_TRUE(ctx.original_game != nullptr);
    SPIEL_CHECK_TRUE(ctx.roots != nullptr);
    const Player non_res = 1 - ctx.resolving_player;
    SPIEL_CHECK_TRUE(ctx.reach_probs[ctx.resolving_player] != nullptr);
    SPIEL_CHECK_TRUE(ctx.cfvs[non_res] != nullptr);

    auto gadget_roots = BuildSubgameRoots(*ctx.roots, non_res,
                                          *ctx.reach_probs[ctx.resolving_player]);
    if (gadget_roots.empty()) return nullptr;
    return CreateGadgetGame(ctx.original_game, std::move(gadget_roots), non_res,
                            *ctx.cfvs[non_res]);
  }

  std::string SubgamePrefix() const override { return "gadget_F:subgame:"; }
  const char* Name() const override { return "Resolving"; }
};

class MaxMarginGadgetStrategy final : public Gadget {
 public:
  std::shared_ptr<const Game> Build(const GadgetContext& ctx) const override {
    SPIEL_CHECK_TRUE(ctx.original_game != nullptr);
    SPIEL_CHECK_TRUE(ctx.roots != nullptr);
    const Player non_res = 1 - ctx.resolving_player;
    SPIEL_CHECK_TRUE(ctx.reach_probs[ctx.resolving_player] != nullptr);
    SPIEL_CHECK_TRUE(ctx.cfvs[non_res] != nullptr);

    auto gadget_roots = BuildSubgameRoots(*ctx.roots, non_res,
                                          *ctx.reach_probs[ctx.resolving_player]);
    if (gadget_roots.empty()) return nullptr;
    return CreateMaxMarginGadgetGame(ctx.original_game, std::move(gadget_roots),
                                     non_res, *ctx.cfvs[non_res]);
  }

  std::string SubgamePrefix() const override { return "mm_F:subgame:"; }
  const char* Name() const override { return "MaxMargin"; }
};

class ResolvingByISGadgetStrategy final : public Gadget {
 public:
  std::shared_ptr<const Game> Build(const GadgetContext& ctx) const override {
    SPIEL_CHECK_TRUE(ctx.original_game != nullptr);
    SPIEL_CHECK_TRUE(ctx.roots != nullptr);
    const Player non_res = 1 - ctx.resolving_player;
    SPIEL_CHECK_TRUE(ctx.reach_probs[ctx.resolving_player] != nullptr);
    SPIEL_CHECK_TRUE(ctx.cfvs[non_res] != nullptr);
    auto gadget_roots = BuildSubgameRoots(*ctx.roots, non_res,
                                          *ctx.reach_probs[ctx.resolving_player]);
    if (gadget_roots.empty()) return nullptr;
    return CreateResolvingByISGame(ctx.original_game, std::move(gadget_roots),
                                   ctx.resolving_player, *ctx.cfvs[non_res]);
  }

  std::string SubgamePrefix() const override { return "ribis_F:subgame:"; }
  const char* Name() const override { return "ResolvingByIS"; }
};

class FullGadgetStrategy final : public Gadget {
 public:
  FullGadgetStrategy(
      FullGadgetGame::Mode mode,
      std::shared_ptr<const Policy> trunk_policy,
      std::unordered_map<std::string, std::vector<std::string>>
          all_boundary_states_by_group)
      : mode_(mode),
        trunk_policy_(std::move(trunk_policy)),
        all_boundary_states_by_group_(std::move(all_boundary_states_by_group)) {}

  std::shared_ptr<const Game> Build(const GadgetContext& ctx) const override {
    SPIEL_CHECK_TRUE(ctx.original_game != nullptr);
    SPIEL_CHECK_TRUE(trunk_policy_ != nullptr);
    return CreateFullGadgetGame(
        ctx.original_game, trunk_policy_, ctx.resolving_player, ctx.pub_obs,
        all_boundary_states_by_group_, mode_,
        /*boundary_portfolios_p0=*/{}, /*boundary_portfolios_p1=*/{},
        /*enumerate_boundary_portfolios=*/true);
  }

  std::string SubgamePrefix() const override { return "full_F:subgame:"; }
  const char* Name() const override { return "Full"; }

 private:
  FullGadgetGame::Mode mode_;
  std::shared_ptr<const Policy> trunk_policy_;
  std::unordered_map<std::string, std::vector<std::string>>
      all_boundary_states_by_group_;
};

}  // namespace

std::unique_ptr<Gadget> MakeUnsafeGadget() {
  return std::make_unique<UnsafeGadgetStrategy>();
}

std::unique_ptr<Gadget> MakeResolvingGadget() {
  return std::make_unique<ResolvingGadgetStrategy>();
}

std::unique_ptr<Gadget> MakeMaxMarginGadget() {
  return std::make_unique<MaxMarginGadgetStrategy>();
}

std::unique_ptr<Gadget> MakeResolvingByISGadget() {
  return std::make_unique<ResolvingByISGadgetStrategy>();
}

std::unique_ptr<Gadget> MakeFullGadget(
    FullGadgetGame::Mode mode,
    std::shared_ptr<const Policy> trunk_policy,
    std::unordered_map<std::string, std::vector<std::string>>
        all_boundary_states_by_group) {
  return std::make_unique<FullGadgetStrategy>(
      mode, std::move(trunk_policy), std::move(all_boundary_states_by_group));
}

}  // namespace open_spiel
