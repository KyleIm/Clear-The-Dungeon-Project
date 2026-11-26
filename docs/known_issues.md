# Known issues and verification limits

These observations refer to the Ren'Py 1.00 source revision in [game_config.json](../data/game_config.json). The static datasets preserve the source values and conditions. Runtime reproduction and the relationship between this source and the shipped Steam build remain unverified.

| Observation | Source reference and consequence |
|---|---|
| Reroll variables are used before their global declarations. | `Char.reroll_stats`, script lines 1661–1684. Standard Python rejects that declaration order. The data records the apparent price update and split algorithm, with actual Ren'Py behavior unverified. |
| Rogue Imitation restore has a local-name hazard. | `Char.process_status_effects`, lines 522–557. The function reads `player_copied_stat` on its player branch and later assigns to it without declaring it global. Standard Python's local binding would cause an unbound-local error when that read is reached. |
| Imitating Ella may fail to find a registry entry. | `copy_from_monster`, lines 439–471, directly looks up the target name in the normal monster registry. Ella's decks are constructed separately and the registry has no Ella entry. |
| Cure handlers remove list elements during iteration. | `process_status_effects`, lines 522–630, can skip following entries when an earlier entry is deleted. This also affects the immediate-cure path. |
| Ella's floor-35 conditional action is overwritten. | `monster_action_selection`, lines 2366–2391, first evaluates floor 35, then independently checks floor ≥75 with an else. The else picks uniformly on floor 35, with an extra RNG draw when the early branch also draws. Floor ≥75 has a repeated, unreachable status condition. |
| An upgrade compares a misspelled key. | `monster_action_selection`, lines 2347–2365, uses the token `bonespear` for its ≥61 match. The actual registry key is `bone spear`. The invalid token is documented as source evidence and is excluded from skill references. |
| Blessed Horn and some status text disagree with code. | Registry line 184 and handler lines 1288–1301. Blessed Horn is accuracy 100 and increases attack, while its description refers to accuracy 90 and an accuracy increase. Stun blocking uses a strict <50 comparison (49%). Discharge subtracts attack without an explicit lower bound (lines 1226–1236). |
| Second-hit routing is split across handlers. | `use_skill` computes second-hit damage, but the action wrappers subtract it from the opposing body even if a substitute survived the first hit. See lines 676–697, 1396–1413, 2910–2943 and 3306–3333. Each wrapper also uses the last returned damage rather than summing hits for Counter. |
| Reroll and other stat changes do not share one global cap. | The reroll assigns its segment lengths directly to base and current stats. Reward and skill paths use their own ACC/EVA checks. The Floor 5 bonus can also exceed a skill's normal cap. See lines 779–793, 1530–1544, 1661–1684 and 3784–3815. |
| Current type and base type gate different shop actions. | The shop screen checks current type and hides type potions when it is Normal. The purchase method separately compares the requested type with base type. See `custom_screen.rpy` lines 390–401 and `Char.buy_type_change` lines 1744–1751. |

The exact Steam build behavior, engine-level RNG seeds, reproducibility across saved-game reloads and any differences caused by compiled game artifacts are unverified. Source-only comparisons in Phase 1 do not establish a real-game clear rate.
