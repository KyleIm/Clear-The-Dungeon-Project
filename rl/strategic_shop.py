"""Strategic shop drafts translated into ordinary simulator shop actions."""

import math

PARENTS = {
    "skills": "Skill Configuration",
    "costs": "Skill Cost Reduction",
    "stats": "Stat Redistribution",
    "merchandise": "Merchandise Purchase",
    "leave": "Leave Shop",
}
MENUS = tuple(PARENTS)[:-1]
STAT_KEYS = ("attack", "defense", "hp", "speed")
LOCK = "lock"
MODES = (None, "skills", "uses", "stats", "costs", "merchandise")


def strategic_catalog(env, transaction_actions):
    mana = env.data.config["mana"]
    limits = mana["max"] + len(env.potions) * mana["potion_increment"]
    return [
        *(("shop_strategy", name) for name in PARENTS),
        ("draft_keep",),
        *(("draft_skill", skill) for skill in env.catalog_skills),
        *(("draft_use", uses) for uses in range(limits + 1)),
        *(("draft_stat_digit", digit) for digit in range(10)),
        ("draft_stat_max",),
        ("draft_stat_keep",),
        *transaction_actions,
    ]


def snapshot(state):
    player = state["player"]
    slots = []
    for skill in player["current_skills"]:
        locked = skill == LOCK
        slots.append(dict(skill=None if locked else skill, locked=locked,
                          uses=0 if locked else player["memorized_skills"].get(skill, 0),
                          mana_cost=0 if locked else player["skill_costs"][skill]))
    used = sum(slot["uses"] * slot["mana_cost"] for slot in slots)
    return dict(slots=slots, unlocked_slots=sum(not slot["locked"] for slot in slots),
                mana_used=used, mana_remaining=player["max_mana"] - used,
                max_mana=player["max_mana"],
                stats={name: player[f"base_{name}"] for name in STAT_KEYS},
                available_points=player["available_points"], gold=state["gold"],
                type=player["type"],
                extra_stats={name: player[name] for name in ("EVA", "ACC", "DR")},
                purchases=sorted(player["purchases"]))


class StrategicShop:
    """One visit ledger and a forward-only draft with no engine mutations until commit."""

    def __init__(self, epsilon):
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or epsilon < 0:
            raise ValueError("shop_repeat_epsilon must be finite and nonnegative")
        self.epsilon = epsilon
        self.reset()

    def reset(self):
        self.counts = dict.fromkeys(MENUS, 0)
        self.strategic_actions = self.engine_actions = 0
        self.mode = None
        self.before = None
        self.gear = []
        self.uses = []
        self.stats = {}
        self.prefix = 0
        self.digits = 0
        self.mana_left = 0
        self.points_left = 0

    def transactions(self, env, state, menu):
        # Reuse public affordability and availability rules without changing the engine screen.
        view = dict(state, shop_menu=menu)
        return [action for action in env._primitive_public_actions(view) if action != ("shop_back",)]

    def actions(self, env, state):
        if self.mode is None:
            available = {"skills", "stats", "leave"}
            available.update(menu for menu in ("costs", "merchandise") if self.transactions(env, state, menu))
            return [("shop_strategy", name) for name in PARENTS if name in available]
        if self.mode in {"costs", "merchandise"}:
            return self.transactions(env, state, self.mode)
        if self.mode == "skills":
            originals = [slot["skill"] for slot in self.before["slots"] if not slot["locked"]]
            learned = set(state["player"]["learned_skills"]) & set(env.catalog_skills)
            if learned.difference(originals):
                candidates = sorted(learned.difference(self.gear))
            else:
                candidates = [originals[len(self.gear)]]
            return [("draft_skill", skill) for skill in candidates] + ([("draft_keep",)] if not self.gear else [])
        if self.mode == "uses":
            cost = state["player"]["skill_costs"][self.gear[len(self.uses)]]
            return [("draft_use", count) for count in range(self.mana_left // cost + 1)]
        if self.mode == "stats":
            place = self.place()
            result = [("draft_stat_digit", digit) for digit in range(10)
                      if self.prefix + digit * place <= self.points_left]
            result.append(("draft_stat_max",))
            name = STAT_KEYS[len(self.stats)]
            original = self.before["stats"][name] - env.shop["redistribution_minimum"]
            if 0 <= original <= self.points_left:
                result.append(("draft_stat_keep",))
            if not self.stats and not self.prefix and self.digits == 0:
                result.append(("draft_keep",))
            return result
        raise RuntimeError(f"Unknown shop draft: {self.mode!r}")

    def place(self):
        return 10 ** (len(str(self.points_left)) - self.digits - 1)

    def visit(self, *, left):
        repeated = sum(max(0, count - 1) for count in self.counts.values())
        return dict(strategic_actions=self.strategic_actions, engine_actions=self.engine_actions,
                    left=left, repeated=repeated, penalty=-self.epsilon * repeated,
                    outcome="leave" if left else "timeout", selections={PARENTS[k]: v for k, v in self.counts.items()})

    def step(self, env, state, action):
        info = {}
        engine_steps = 0

        def apply(primitive):
            nonlocal state, engine_steps
            state = env.game.step(primitive)
            engine_steps += 1
            self.engine_actions += 1

        penalty = 0.0
        verb, *args = action
        if verb == "shop_strategy":
            menu = args[0]
            self.strategic_actions += 1
            if menu == "leave":
                apply(("leave_shop",))
                info["shop_selection"] = dict(function=PARENTS[menu], repeat=False)
                info["shop_visit"] = self.visit(left=True)
                self.reset()
            else:
                self.counts[menu] += 1
                repeat = self.counts[menu] > 1
                penalty = -self.epsilon if repeat else 0.0
                info["shop_selection"] = dict(function=PARENTS[menu], repeat=repeat)
                self.mode = menu
                self.before = snapshot(state)
                if menu == "skills":
                    self.mana_left = self.before["max_mana"]
                if menu == "stats":
                    minimum = env.shop["redistribution_minimum"]
                    self.points_left = (sum(self.before["stats"].values()) + self.before["available_points"]
                                        - minimum * len(STAT_KEYS))
        elif self.mode in {"costs", "merchandise"}:
            apply(("shop_menu", self.mode))
            apply(action)
            apply(("shop_back",))
            info["shop_change"] = self.change(self.mode, state, item=args[0] if args else verb)
            self.clear_draft()
        elif verb == "draft_keep":
            info["shop_change"] = self.change(self.mode, state)
            self.clear_draft()
        elif self.mode == "skills":
            self.gear.append(args[0])
            if len(self.gear) == self.before["unlocked_slots"]:
                self.mode = "uses"
        elif self.mode == "uses":
            value = args[0]
            skill = self.gear[len(self.uses)]
            self.uses.append(value)
            self.mana_left -= state["player"]["skill_costs"][skill] * value
            if len(self.uses) == len(self.gear):
                self.commit_skills(env, state, apply)
                state = env.game.get_state()
                info["shop_change"] = self.change("skills", state)
                self.clear_draft()
        elif self.mode == "stats":
            minimum = env.shop["redistribution_minimum"]
            name = STAT_KEYS[len(self.stats)]
            pending = self.points_left if verb == "draft_stat_max" else self.before["stats"][name] - minimum
            if verb == "draft_stat_digit":
                place = self.place()
                self.prefix += args[0] * place
                self.digits += 1
                if place > 1:
                    return state, penalty, dict(info, engine_steps=0, shop_penalty=penalty)
                pending = self.prefix
            self.stats[name] = minimum + pending
            self.points_left -= pending
            self.digits = self.prefix = 0
            if len(self.stats) == len(STAT_KEYS):
                self.commit_stats(env, apply)
                state = env.game.get_state()
                info["shop_change"] = self.change("stats", state)
                self.clear_draft()
        return state, penalty, dict(info, engine_steps=engine_steps, shop_penalty=penalty)

    def change(self, menu, state, *, item=None):
        after = snapshot(state)
        return dict(function=PARENTS[menu], before=self.before, after=after,
                    item=item, gold_spent=self.before["gold"] - after["gold"])

    def clear_draft(self):
        self.mode = None
        self.before = None
        self.gear = []
        self.uses = []
        self.stats = {}
        self.prefix = self.digits = self.mana_left = self.points_left = 0

    def commit_skills(self, env, state, apply):
        player = state["player"]
        slots = [slot for slot, skill in enumerate(player["current_skills"]) if skill != LOCK]
        current = [player["current_skills"][slot] for slot in slots]
        desired_uses = dict(zip(self.gear, self.uses))
        if current == self.gear and all(player["memorized_skills"][skill] == count
                                        for skill, count in desired_uses.items()):
            return
        apply(("shop_menu", "skills"))
        learned = set(player["learned_skills"]) & set(env.catalog_skills)
        while current != self.gear:
            diffs = [i for i, (old, new) in enumerate(zip(current, self.gear)) if old != new]
            available = [i for i in diffs if self.gear[i] not in current]
            if available:
                index = available[0]
                new = self.gear[index]
            else:
                index = diffs[0]
                # A temporary unequipped skill breaks a permutation cycle via legal replacements.
                new = min(learned.difference(current))
            apply(("replace_skill", current[index], new))
            current[index] = new

        uses = env.game.player.memorized_skills.copy()
        for skill in self.gear:
            for _ in range(max(0, uses[skill] - desired_uses[skill])):
                apply(("forget", skill))
        for skill in self.gear:
            for _ in range(max(0, desired_uses[skill] - uses[skill])):
                apply(("memorize", skill))
        apply(("shop_back",))

    def commit_stats(self, env, apply):
        # Release points from reductions before spending them on increases.
        if self.stats == self.before["stats"]:
            return
        apply(("shop_menu", "stats"))
        deltas = {name: self.stats[name] - self.before["stats"][name] for name in STAT_KEYS}
        increments = sorted(env.shop["redistribution_steps"], reverse=True)
        for sign in (-1, 1):
            for name, delta in deltas.items():
                remaining = sign * delta
                for increment in increments:
                    while remaining >= increment:
                        apply(("adjust_stat", name, sign * increment))
                        remaining -= increment
        apply(("shop_back",))

    def features(self, env, state, scale, one_hot):
        in_shop = state["phase"] == "shop"
        current = snapshot(state) if in_shop else None
        before = self.before if in_shop and self.before else current
        result = [scale(self.counts[name], 10) if in_shop else 0.0 for name in MENUS]
        result.extend(one_hot(self.mode if in_shop else None, MODES))
        result.extend((scale(len(self.gear), 6), scale(len(self.uses), 6),
                       scale(len(self.stats), 4), scale(self.prefix, 100), scale(self.digits, 10),
                       scale(self.mana_left, 170), scale(self.points_left, 100)))
        position = 0
        for i in env.slots:
            original = before["slots"][i] if before else None
            name = original["skill"] if original else None
            locked = bool(original and original["locked"])
            result.extend(one_hot(LOCK if locked else name, env.skill_tokens))
            result.extend((scale(original["uses"], 10) if original else 0,
                           scale(original["mana_cost"], 10) if original else 0))
            if not locked and position < len(self.gear):
                draft = self.gear[position]
                draft_uses = self.uses[position] if position < len(self.uses) else 0
                result.extend(one_hot(draft, env.skill_tokens))
                result.extend((1.0, float(position < len(self.uses)), scale(draft_uses, 10),
                               scale(state["player"]["skill_costs"][draft], 10)))
            else:
                result.extend(one_hot(LOCK if locked else None, env.skill_tokens))
                result.extend((0.0, 0.0, 0.0, 0.0))
            if not locked:
                position += 1
        result.extend(scale(before[name], 170) if before else 0.0 for name in
                      ("max_mana", "mana_used", "mana_remaining", "unlocked_slots"))
        for name in STAT_KEYS:
            result.extend((scale(before["stats"][name], 100) if before else 0.0,
                           float(name in self.stats), scale(self.stats.get(name, 0), 100)))
        result.append(scale(before["available_points"], 100) if before else 0.0)
        result.append(scale(before["gold"], 1000) if before else 0.0)
        result.append(scale(state["reroll_cost"], 100) if in_shop else 0.0)
        result.extend(scale(env.shop["prices"][item], 1000) if in_shop else 0.0 for item in
                      (*env.items, "skill_cost_reduction", "type_change"))
        return result
