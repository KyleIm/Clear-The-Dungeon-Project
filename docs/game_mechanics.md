# Dungeon & Rogue game mechanics

This static reference describes the Ren'Py 1.00 source revision recorded in [game_config.json](../data/game_config.json). The private source is needed for source comparison. Runtime behavior in the shipped Steam build is unverified. See [known issues](known_issues.md) for source anomalies and their expected consequences.

## Dataset conventions

- [Skills](../data/skills.json) lists every registry entry in source order. Skill names are the original registry keys. Numeric IDs are stable one-based ordinals for this snapshot because the game has no separate numeric skill ID.
- [Monsters](../data/monsters.json) keeps the original monster indices and ordered skill names. Ella skill variants describe eligible names and slot construction because her deck is assembled at encounter time. AI slot indices are zero-based.
- [Areas](../data/areas.json) uses monster IDs for encounter lists. Region and skill names retain source spelling, including `Phoneix`. HP in the base-stat block is half the starting maximum HP.
- [Game configuration](../data/game_config.json) holds the starting conditions, combat constants, random sampling settings, and economy. The datasets describe game rules and contain no saved runs. Future experiment results belong in SQLite.

## Run setup and floor progression

The Rogue starts on floor 1 in area 1 with base attack, defense, HP, and speed of 100 each, zero additional stats, Normal type, zero gold, and 200 current HP. `area_selection` shuffles the remaining nine regions and removes the selected region. On the first region, `Char.initial_skill` assigns the regional starter, two attacks from one shuffled attack pool, and a skill from the shuffled rank pool. The first Ella uses the paired regional starter, the next two entries from the same attack shuffle, and the next rank skill. Starting player uses are 20/10/10/10 in slot order.

| Floor condition | Encounter or transition |
|---|---|
| 1–90 ending in 1 | Fixed regional entry monster |
| 1–90 ending in 7 | Fixed regional midboss |
| 1–90 ending in 0 | Fixed regional stage boss, added to the Gate queue on entry |
| Other ordinary floors | Uniform draw from that region's ordered pool. Below floor 10 the pool is truncated to the recorded size. |
| 5, 35, 75 | Ella replaces the ordinary encounter. The path chosen after floor 35 selects her floor-75 variant. |
| 91–99 | The Gate replays the nine stage bosses in initial encounter order. Each boss is reset and healed. |
| 99 after the Gate boss | Path 1 adds another Ella battle on floor 99. Path 2 heals, resets and restores saved skill uses and opens a shop. |
| 100 | Lair and Dragon. |

The ordinary post-victory and post-shop spawn paths set `T = 5 × floor(floor_number / 10)`. From two independent inclusive integer draws in `[0,T]`, sort the two cuts as `lo, hi`. Add `lo` to accuracy, `hi − lo` to evasion, and `T − hi` to damage reduction. The floor-1 entry and the direct stage boss, Gate, Ella and Dragon paths bypass the sampling code. After the stage boss reward, the player recovers to full HP, resets battle stats, restores saved uses, and enters the shop. Shop exit advances the floor and area, then sets up the next encounter.

After defeating Ella on floor 5 the player receives current-stat increases for the region. Fixed fractions are floored against the named base stat.

| Region | Bonus |
|---|---|
| Forest, Highland | Attack +3/4 |
| Fort | Evasion +50 and attack +1/4 |
| Graveyard | Evasion +50 and defense +1/4 |
| Mansion | Attack +1/2 and defense +1/4 |
| Party, Spiritland | Speed +1/4 and defense +1/2 |
| Ruin | Defense +3/4 |
| Temple | Accuracy +50 and speed +1/4 |

After defeating Ella on floor 35, pickpocketing sets path 1 and helping sets path 2. Both assign the player's base and current type to Ella's current type.

## Turn order

1. The player selects an equipped skill with at least one use. When all unlocked skills have exactly zero uses, Sacrifice is offered. Selection records the action in both the selected-action and last-selected-action fields.
2. The monster constructs its available list, chooses an action and records it. Priority is +1 for categories 1, 52 and 57, −1 for 5 and 50, and 0 otherwise.
3. Compare priority and then current speed, higher first. A complete tie uses a uniform draw. The first actor skips its move if its priority is −1. The second actor is called only while both main combatants have positive HP.
4. For each attempted move the action wrapper checks that actor's status effects in list order. If allowed to act, `Char.use_skill` spends one equipped use for the Rogue before resolution. Sacrifice and the action called by the Rogue's Mimic bypass this charge. Monsters choose from their available list each turn. A miss still spends a Rogue use.
5. Resolve damage or skill effects, including wrapper-side follow-up hits, counter, Mimic, wake-up, on-hit statuses and Gold Rush. A living substitute can receive the initial target-directed effect. The wrappers and individual skills have exceptions described below.
6. Reset priorities, process end-of-turn status effects for player then enemy, clear both last-damage fields, save, and check terminal HP.

## Hit and damage calculations

Where a standard hit roll applies, draw an integer uniformly from 1 through 100 inclusive. Hit if the roll is at or below `skill accuracy + caster accuracy − actual target evasion`. In branches that check the target mark, replace the threshold with 100 when the actual target is marked. There is no explicit threshold clamp. Raw accuracy fields can exceed 100 and are thresholds in this formula. Release uses its own accuracy plus caster accuracy with no target evasion term. Some self effects and Sacrifice make no hit roll. See each skill's condition field for its branch.

Ordinary formula damage is:

```text
raw = power × numerator_stat / (target defense × aurora divisor)
damage = max(0, int(raw × type multiplier) − target damage reduction)
```

Attack is the usual numerator. Jiujitsu uses the actual target's attack, Flash Beak uses caster speed, and Shield Slam uses caster defense. Aurora multiplies the divisor by 1.5. Matching caster and skill types multiply damage by 1.2. Beast beats Elemental, Elemental beats Undead, and Undead beats Beast. An advantage multiplies by 1.5, a disadvantage by 2/3. The caster's reverse status swaps those outcomes. Judgement also counts Undead as favorable before reverse. The ordinary damage branches cap applied damage at remaining HP.

## Individual skills and exceptions

Every one of the 88 skills has an English effect and activation condition in [skills.json](../data/skills.json). Each entry also points to its definition and category handler by line number in the pinned `game/script.rpy`. Power can be a fixed damage amount, a stat divisor, or a parameter. The source handler determines its meaning.

| Skills or categories | Resolution details |
|---|---|
| Ordinary and priority attacks (0–1) | Use the ordinary damage formula. The priority variant modifies move order. |
| Double Strike, Magic Arrow, Bonemerang (2) | A second roll follows a successful first roll and uses the same threshold. Damage can be recalculated after a substitute falls. The wrapper subtracts the second returned amount from the body. |
| Pierce, Flesh Explosion, Firebird (3), Flame Wheel (45), Sacrifice (99) | Recoil is half actual damage rounded down, blocked by helmet. Flame Wheel is fixed damage. Sacrifice is floor(caster base HP / 2) fixed damage without a roll. Rage (46) instead recoils on a miss for half the uncapped ordinary formula result, unless helmet is active. |
| Drain, Kiss (4), Blood Leech (42) | Heal from actual damage, half rounded down for category 4 and full for Blood Leech. Healing is capped at maximum HP. |
| Counter (5), Mimic (50) | Counter makes a hit roll before the wrapper applies recorded damage plus half rounded down. A protecting substitute prevents counter damage. Mimic invokes the opponent's recorded selected skill with no separate roll. It refuses Counter or Mimic and runs the normal action wrapper again. |
| Elemental Orb (6), Despair (7), Deathscythe (41), Acid Stone (56), Hybrid Attack (57) | Fixed power, half current target HP removed, HP set to 1, power minus DR with DR −5, and two priority fixed hits respectively. Hybrid Attack shares the wrapper's second-hit routing. |
| Rank and status skills (10–39) | Attack, defense and speed rank deltas generally use base stat divided by power. Direct accuracy/evasion rank effects use fixed additions or subtractions with branch-specific ±80 bounds. Opponent pure debuffs and directed statuses are blocked by a substitute. Release, Magician Hat, Heart of Lion, Rigid, Molt and Sap Screw use their individual rules in the registry. |
| Sturdy Kick, Will of Wisp, Discharge, Blessed Horn, Blindfist | Apply ordinary damage before their stat change. Will of Wisp skips its accuracy debuff on a substitute. Blessed Horn raises caster attack and uses registry accuracy 100. |
| Trick or Treat (47), Gold Rush (55) | Trick or Treat uniformly selects another non-Normal current type. Gold Rush pays the player floor × 100 on a hit in either wrapper. |
| Icebolt, Poison Tail, Stinger (58–60) | The wrapper applies the selected status. Icebolt and Poison Tail have a separate 30% roll after the hit when the target is a body without the corresponding status or immunity. The recipient also must survive. Stinger deals fixed damage and uniformly picks stun or poison if the surviving body has neither and lacks immunity. |
| Heal, Recovery (61–62) | Heal restores up to floor(maximum HP / power). Recovery calls the immediate-cure routine and can be used through stun and sleep. |
| Imitation, Revive, Breath (96–98) | Imitation copies applicable attributes and a skill list. Revive costs 20 HP and leaves a substitute with 20 HP only when the caster can pay and survive and its substitute is absent. Breath consumes the hit roll before checking caster HP ≤ half maximum. |

A substitute starts with base attack, defense and speed 1, base HP 0, and zero additional stats. Its current HP is set to 20 when created, with body/depleted checks in the skill handlers. Imitation copies base and current attack, defense, accuracy, evasion and damage reduction, plus current type. It copies neither HP nor speed. The Rogue's Imitation looks up the target by name in the original monster registry and assigns five uses per copied skill. The monster version copies every unlocked player slot and has its own temporary skill list.

## Status effects

Status application appends an entry with a zero cure counter. Actor-side effects run as part of checking for a blocked action, so ordering and early returns matter. Poison and sprout subtract at most HP minus 1. Sprout also heals the opponent by the deducted amount. Regeneration and drains are capped to maximum HP.

| Status | Effect and removal |
|---|---|
| Stun | Roll < 50 on 1–100 before acting, for 49% action denial. Recovery bypasses denial. At turn end, remove on a roll ≤ counter or add 16.5 after failure. |
| Poison | Lose floor(max HP / 8) before acting. Roll ≤ counter to remove at turn end, otherwise add 16.5. |
| Sprout | Transfer floor(max HP / 10), limited to leave 1 HP and subject to recipient healing cap. Roll ≤ counter to remove, otherwise add 25. |
| Regeneration | Heal floor(max HP / 8) before acting. First failed equality roll changes counter 0 to 25. Later remove only on an exact roll of 25. |
| Sleep | Block the action except Recovery. Some hit wrapper branches call wake-up directly. At turn end, roll ≤ counter to wake, otherwise add 20. |
| Target | Checked attack branches force threshold 100. At turn end remove when counter is 100, otherwise add 100. |
| Reverse | Invert the caster's attack type advantage. Remove at 100, otherwise add 20 at turn end. |
| Aurora | Divide ordinary incoming damage by 1.5. Remove at 100, otherwise add 25. |
| Immunity | Individual application branches check it, including beneficial statuses that check it on self. Remove at 100, otherwise add 20. |
| Helmet | Prevents recoil in the skills that test it. Remove at 99, otherwise add 33. |
| Imitation | Restores backed-up fields through the immediate-cure path only, subject to the limitations in the issue list. |

The source removes from the list it is iterating. Several effects being removed together can therefore produce different results from a full list clear.

## Enemy action choice

Use the copied skill list during Imitation. Otherwise use the ordered monster registry or the constructed Ella deck. Starting at floor 31 the base starters upgrade to Bone Spear, Bear Hand and Wave. At 61 or above the source uses a separate mapping to Undead Missile, Tiger Claw and Tornado, including a misspelled Bone Spear comparison. The mapping changes only slot 0.

Most enemies choose uniformly among their available skills. [Monsters](../data/monsters.json) records the ordered conditional rules for Lionheart, Ogre, Tarrasque, Queen, Superminion, Vampire, Grimreaper, Dragon and Ella. The floor-35 Ella branch chooses twice when its early branch also uses a random choice. At 75 and 99 the status check precedes the speed/full-HP check and the remaining HP-based choices. Store the resulting skill as the enemy's recorded action before initiative is computed.

## Rewards and shop

After a win below floor 100, choose a skill uniformly from the enemy's last available list filtered to entries missing from the player's learned deck. If available, the final choice is learned independently of the player's reward menu choice. Exchange draws again from the same eligible list, including the first draw. The other rewards are +1000 gold, full HP, current-stat and status reset, or a uniformly selected +1 to base and current evasion, accuracy or damage reduction. The two 80-point bounds are checked separately for base and current stat growth. Damage reduction has no equivalent cap.

Mana is a pre-battle allocation budget. At the skill-management screen, add or remove skill uses within the budget `max mana − sum(cost × allocated uses)`. A learned, unequipped skill may replace a selected slot with zero uses. Shop exit copies the use allocation. Entry to the next shop restores it. The two keys cost 1000 and 2000 and fill slots 5 and 6 with the first available unowned-by-slot skill from the learned list. The three sequential mana potions cost 1000, 2000 and 3000 and add 10 each. Red key requires blue. A skill's original cost can be reduced by 1 once for 1000 gold if its cost is above 1. The type-change items cost 500. The shop screen hides them for Normal current type and hides the matching current type. The purchase method compares base type before charging.

Attack, defense, base HP and speed can be redistributed in steps of 1 or 10 with a minimum of 1 per stat. Reset sets all four to 1 and returns the remainder to the pool. Exiting redistribution resets current stats and heals the player. Reroll repartitions the sum of base evasion, accuracy and damage reduction when positive. Sort two independent inclusive draws from 0 to the total, then assign the three segment lengths in evasion, accuracy, damage reduction order. It updates current values too. Price starts at 100, doubles on purchase, and resets when leaving the shop. See the issue list for the eligibility of this source routine to run.

Gameplay selections and probability rolls primarily use `renpy.random`. Reroll uses Python's `random.randint`. No explicit seed setup was identified in the inspected scripts. Engine-level RNG initialization and the save/load relationship of either random stream remain unverified. Two sorted cuts do not give equal probability to every integer partition.

## Run termination

After the turn finishes, the decision handler tests player HP first. Player HP ≤ 0 is game over, including a simultaneous knockout. Enemy HP ≤ 0 with a living player is a victory. At floor 100 this is the Dragon win and goes directly to the ending and credits without a reward menu. Game over and completion of the credits attempt to delete save slot `1-1`. The source saves to that slot at action, turn, and certain progression boundaries. Rollback is disabled.

## Validation

Run `python tests/validate_game_data.py` from the project root. For source comparison, append `--source-script <path-to-private-checkout>/game/script.rpy` using the revision in the configuration. The optional comparison reads source as text and uses static parsing. It does not run the game, require credentials, or copy the private files into this repository.
