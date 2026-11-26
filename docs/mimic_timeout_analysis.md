# Mimic timeout analysis

All 87 action-limit timeouts in the [million-game random baseline](baseline_random_1m_results.md) reproduced against Mimic at the recorded floor and reached the cap of 10,000 actions with both combatants alive. The original Dungeon & Rogue 1.00 Ren’Py combat code reproduced the same deadlock. Counter remains selectable because its execution is skipped before it can spend a use. Mimic refuses to copy Counter, and the positive Counter count keeps Sacrifice unavailable.

## SQLite extraction and seeded replay

The existing `data/baseline_random_1m/baseline.sqlite3` was opened read-only. The replay used the simulator at `2187d2a817376cb6080983d86e534551d6d80536` and the baseline runner’s action selection. Each episode called `Game.reset(game_seed)` and used a separate `random.Random(policy_seed)` to choose uniformly from `get_legal_actions()` before each `Game.step()`. It executed the recorded number of actions. The action count includes choices outside combat.

```sql
SELECT game_id, game_seed, policy_seed, floor, actions
FROM episodes
WHERE outcome = 'timeout' AND detail = 'action_limit'
ORDER BY game_id;
```

| Floor at timeout | Reproduced episodes |
|---|---:|
| 2 | 57 |
| 3 | 19 |
| 4 | 9 |
| 6 | 2 |
| **Total** | **87** |

The local tracing harness recorded **869,868 combat turns** across the 87 episodes. Each turn contains the legal actions, the chosen action, both actors’ HP, status effects, speed, substitute HP, `selected_action` and `ram_action`, the player’s equipped slots and remaining uses, and the order and HP results of the actual resolver calls. The compressed `game_<game_id>.jsonl.gz` traces and per-episode `summary.json` are in the local `mimic-timeout-traces` directory beside the project checkout. The per-episode summary contains both seeds, the onset action and the last traced turn.

## Repeating combat state

Counter became the only legal action at episode action **50–90**. The first turn that returned an identical state began one action later in every case, at episode action **51–91**, corresponding to turn **35–86** of the Mimic encounter. The first forced Counter turn updated the saved action fields from the preceding skill. From the next turn through the recorded cap, the simulator returned the same `get_state()` values except for the incrementing combat turn counter. The underlying PRNG state can advance because the enemy and policy still make choices from singleton action lists. No outcome-changing random branch was available in the repeating state.

Across all 87 episodes, the verified repeating part covered **864,457 turns**, or **9,910–9,950 turns per episode**. In those states the player had **2–200 HP**, Mimic had **6–200 HP**, and neither actor had an active status effect or substitute. Player speed was 100 and Mimic speed was 20. Every equipped skill except Counter had zero uses. Counter retained 3–10 uses, with 10 in 43 episodes. At the end of every repeated turn the player’s action pair was `(counter, counter)` and Mimic’s was `(mimic, mimic)`, in `selected_action`, `ram_action` order.

### Example turn trace

Game `16774` used game seed `2555044005428608896` and policy seed `866909666696650408`. On floor 2, the Rogue had 180 HP and Mimic had 200 HP throughout the turns below. Both actors had empty status lists and no substitutes. In equipped order, the Rogue’s skills were Thunder, Fast Claw, Counter and Weakness. The last two slots were locked.

| Episode action | Fight turn | Legal actions | Chosen | Remaining uses before → after | Rogue action pair after | Mimic action pair after | Resolver outcome |
|---:|---:|---|---|---|---|---|---|
| 64 | 61 | Thunder, Counter | Thunder | `(1, 0, 10, 0)` → `(0, 0, 10, 0)` | `(thunder, thunder)` | `(thunder, mimic)` | Both Thunder resolutions deal 0 HP damage. |
| 65 | 62 | Counter | Counter | `(0, 0, 10, 0)` → same | `(counter, counter)` | `(mimic, mimic)` | Counter is skipped. Mimic executes but declines to copy. Action pairs change from turn 61. |
| 66 | 63 | Counter | Counter | `(0, 0, 10, 0)` → same | `(counter, counter)` | `(mimic, mimic)` | Same resolver outcome. The state repeats. |
| 67 | 64 | Counter | Counter | `(0, 0, 10, 0)` → same | `(counter, counter)` | `(mimic, mimic)` | The state repeats again. |
| 10,000 | 9,997 | Counter | Counter | `(0, 0, 10, 0)` → same | `(counter, counter)` | `(mimic, mimic)` | Both combatants still alive at the runner’s cap. |

### Why the state persists

1. Mimic’s only skill is Mimic. Counter and Mimic both have priority −1. Higher speed puts the Rogue in the first slot in all 87 encounters.
2. The first slot skips execution for priority −1. The Rogue never enters the action wrapper or reaches the skill-use deduction. The selection remains in `ram_action`.
3. Mimic executes in the second slot while both actors are alive. Its copy branch rejects `ram_action` values `counter` and `mimic`, so it ends without a copied action or damage.
4. The end-of-turn phase leaves HP, uses and action availability unchanged in the observed states. With even one Counter use remaining, the fallback condition still fails and the next turn offers Counter again.

## Sacrifice and the copy branch

The combat menu offers Sacrifice when **all equipped, unlocked skills have zero remaining uses**. None of the 87 seeded replays offered Sacrifice in a Mimic fight, and no Sacrifice reached the skill resolver anywhere in those episodes. Its HP loss was zero in the timeout traces because it never executed.

Four controlled one-turn comparisons started with all equipped uses set to zero and matched the original combat code. With both actors able to act, the Rogue’s Sacrifice executes before priority −1 Mimic. Mimic then copies `sacrifice` into its `selected_action`, leaves its own `ram_action` as `mimic`, and re-enters the enemy action wrapper. A caster’s Sacrifice removes `min(target.hp, caster.base_hp // 2)` from the resolved target, which can be a substitute. It removes `min(caster.hp, damage // 2)` from the caster unless the caster has Helmet. Sacrifice uses no skill charge. Both casters had 100 base HP in these probes, so each uncapped hit removed 50 target HP and caused 25 recoil where Helmet was absent.

| Initial HP (Rogue, Mimic) | Condition | HP after Rogue’s cast | HP after Mimic’s copied cast | Outcome |
|---|---|---|---|---|
| `(200, 200)` | Neither has Helmet | `(175, 150)` | `(125, 125)` | Both casts resolve. |
| `(200, 200)` | Mimic has Helmet | `(175, 150)` | `(125, 150)` | Helmet prevents Mimic’s 25 recoil. |
| `(200, 20)` | Neither has Helmet | `(190, 0)` | — | The hit caps at 20, so Rogue takes 10 recoil. Mimic is already dead. |
| `(20, 200)` | Neither has Helmet | `(0, 150)` | — | Rogue’s recoil caps at its remaining 20 HP. The second action is skipped. |

## Original Ren’Py comparison

The private source reference used Dungeon & Rogue 1.00 commit `480dc17d1354055c783cb8a6a0cea9a22028093e`. A local CPython adapter executed the original Ren’Py combat Python and label control flow with presentation and save calls stubbed. For each replay it initialized the source reference from the first Counter-only turn, with the same actor state and PRNG state. The first **five forced turns in each episode matched**, for **435 turn comparisons** of actor state, status, remaining uses, action fields and game RNG state. The Sacrifice probes compared the combat resolution itself, before reward processing. The Steam executable was outside the verification scope.

| Mechanism | Private source location | Simulator counterpart |
|---|---|---|
| Combat skill availability and both action fields | `custom_screen.rpy:250–274`, `skill_selection` | [`Game.get_legal_actions()` and `_combat_step()`](../simulator/game.py) |
| Mimic’s skill list, speed and action selection | `script.rpy:227, 1968, 2339–2448` | [`enemy_ai.select_action()`](../simulator/enemy_ai.py) |
| Priority, speed tie-break and first-slot skip | `script.rpy:1900–1921, 2450–2524`, `set_priority_level`, `priority_comparison`, `execute_actions` | [`combat.resolve_turn()`](../simulator/combat.py) |
| Skill-use deduction and Mimic’s refusal or recursive copy | `script.rpy:637–652, 1316–1318, 3353–3360`, `Char.use_skill` and `execute_monster_action` | [`combat._execute_action()`](../simulator/combat.py) and [`skills.resolve_skill()`](../simulator/skills.py) |
| Sacrifice damage and recoil | `script.rpy:1507–1515`, `Char.use_skill`, category 99 | [`skills._amount()` and `_resolve_damage()`](../simulator/skills.py) |
| Combat continuation | `script.rpy:3752–3758`, `battle_system` | [`Game._combat_step()`](../simulator/game.py) and the [baseline action cap](../experiments/random_baseline.py) |

**Classification.** The deadlock is an original-game rules defect reproduced by the simulator. The original battle loop continues as long as both combatants have positive HP. The experiment runner’s action cap records that unresolved combat as a timeout.
