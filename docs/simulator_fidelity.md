# Phase 2 Independent Simulator Fidelity Audit

## Baselines and findings

- **Original source baseline.** Dungeon & Rogue 1.00 commit `480dc17d1354055c783cb8a6a0cea9a22028093e` in the private repository supplied the reference behavior. The audit reread `script.rpy`, `custom_screen.rpy`, `screens.rpy`, and `options.rpy` at that commit. Expected results came directly from the source, independent of prior analysis reports or descriptions in the public JSON.
- **Simulator baseline.** The audit used public simulator commit `49b86c8295584735b5fbbb054a3ef76439eb1b6f`. The game engine and original repository stayed unchanged during the audit.
- **Main finding.** Applying or removing enemy Imitation at certain points caused the simulator to offer different **victory skill reward candidates** from the original. Both discrepancies reproduced with identical inputs and random seeds. The tested shop operations for skill replacement, use allocation, and cost reduction reproduced the original state changes. The `forget` action reclaims 1 allocated use and retains the skill.

## 1. Comparison of all available actions

The audit used actionable screen controls and menu branches in the original labels. It distinguishes choices that have an effect, no-op clicks, and screen navigation. The tuples use the format returned by `Game.get_legal_actions()`.

| Action in the original | Simulator action | Comparison |
|---|---|---|
| Cast an equipped skill with an unlocked slot and uses > 0 | `("skill", name)` | Matches. The combat screen blocks casting from locked slots and slots with 0 uses. |
| Use Sacrifice when every unlocked slot has exactly 0 uses | `("skill", "sacrifice")` | Matches. Sacrifice is a fallback for the regular slots and has no allocated uses. |
| Hover over combat info icons or view tooltips in rewards and shops | Some information from `get_state()` | Interface difference. These clicks do not advance combat. The simulator has no dedicated action for opening information. |
| Pickpocket immediately after defeating Ella on floor 35 | `("ella", "pickpocket")` | Route 1 and the timing of the type change match. |
| Help Ella immediately after defeating her on floor 35 | `("ella", "help")` | Route 2 and the timing of the type change match. |
| Choose Gold, Potion, Initialize, Stat, or Exchange after a victory | `("reward", "Gold")`, `"Potion"`, `"Initialize"`, `"Stat"`, `"Exchange"` | All five choices are available. Choosing Exchange advances the game even with no candidates. See §3 for the candidate generation differences involving Imitation. |
| Open skill setup from the shop menu | `("shop_menu", "skills")` | The original refreshes its stored mana balance on entry. The simulator calculates the balance. |
| Press + in skill setup | `("memorize", name)` | Allocates 1 use to the slot. When mana is insufficient, the simulator omits the action and clicking the original button has no effect. |
| Press − in skill setup | `("forget", name)` | Reclaims 1 use from the slot. The skill remains learned and equipped. At 0 uses, the original button can still be clicked and has no effect. |
| Click an equipped skill, then choose a replacement from the inventory | `("replace_skill", old, new)` | Combines the choices from two screens into one action. Only learned, unequipped skills are eligible. The new skill starts with 0 allocated uses. |
| Press Back in the replacement inventory | No corresponding action | Interface difference. The original returns to skill setup without making a change. The simulator has no intermediate state awaiting a replacement. |
| Press Confirm in skill setup | `("shop_back",)` | Returns to the shop menu. |
| Open skill cost reduction from the shop menu, then click a learned skill | `("shop_menu", "costs")` → `("reduce_skill_cost", name)` | Applies to learned, unequipped skills too. The original shows the button when gold is insufficient, the cost has already been reduced, or the cost is ≤1. Clicking has no effect in these cases. |
| Press Confirm in skill cost reduction | `("shop_back",)` | Returns to the shop menu. |
| Open main stat redistribution from the shop menu, then adjust ATK/DEF/HP/SPD by ±1 or ±10 | `("shop_menu", "stats")` → `("adjust_stat", stat, delta)` | Represents 16 combinations across the four stats. The original still allows clicks with no effect when resources are insufficient or an adjustment would go below the minimum. |
| Press Reset in redistribution | `("reset_redistribution",)` | Sets all four base values to 1 and returns the difference to the allocation pool. |
| Press Confirm in redistribution | `("shop_back",)` | Resets current stats and status, restores health, and returns to the shop menu. |
| Open merchandise from the shop menu | `("shop_menu", "merchandise")` | Corresponds to opening the merchandise screen. The original label also saves the game on entry. |
| Press Reroll | `("reroll",)` | The original always shows the button, and its callback includes a save call. The simulator offers the calculation when the gold and bonus stat total meet the requirements. See §3 for the original compilation issue and saving differences. |
| Buy the blue key, then buy the red key | `("buy", "blue_key")`, `("buy", "red_key")` | Price, prerequisites, and slot order match. If the player has sufficient gold and a locked slot but no unequipped skill, both versions deduct the cost as the only effect. |
| Buy the first, second, or third mana potion | `("buy", "blue_mana_potion")`, `"elemental_mana_potion"`, `"undead_mana_potion"` | Purchase order and the +10 maximum mana increase match. The simulator omits purchases that would have no effect in the original because of insufficient gold. |
| Change type to Undead, Elemental, or Beast | `("change_type", target)` | Supports all three targets. The original screen hides the choices when the current type is Normal and hides the option for the current type. The function has no effect when the target equals the base type. The simulator excludes no-op choices. |
| Press Confirm in merchandise | `("shop_back",)` | Returns to the shop menu. |
| Select Enter next area | `("leave_shop",)` | Matches the allocation copy, reroll price reset, floor and area counter increments, and preparation of the next encounter. |
| Select Start and confirm overwriting an existing save | `reset(seed)` / launch the terminal | Starts a new game. Save overwrite confirmation is missing. The seed argument is an extra simulator feature. |
| Select Load from the start menu or return to the main menu | No direct equivalent | Missing functionality. The simulator cannot load the original save slot or resume from the main menu. The terminal's `q` exits the process. |
| Advance dialogue or credits, use Skip/Auto, History, Options, display, sound, language, About, Badges, or Return / Quit | Terminal prompts and `q`, with no other direct equivalents | Interface difference. These are rendering, dialogue, and settings controls. See §3 for whether the original menu and rollback controls are reachable. |

Source references for combat are `custom_screen.rpy:250–274` and `script.rpy:2334–2337`, for Ella `script.rpy:3823–3870`, for rewards `custom_screen.rpy:464–476` and `script.rpy:4013–4067`, and for shops `custom_screen.rpy:276–460` and `script.rpy:4244–4369`. The system menu review covered `screens.rpy:294–299,416–445,470–525,646–676,780–870,924–996,1884–1900` and `options.rpy:15–23`. The corresponding simulator code is the action list and `step` in `simulator/game.py`, `shop.py`, `rewards.py`, and the terminal adapter.

The audit found no movement, wait, flee, direct item use, permanent skill removal, or area selection menu in ordinary combat. The simulator has no standalone actions for these either. Explicit seeding, JSON list actions, full state observation, and policy execution without interactive input are programmatic interfaces unique to the simulator. Structured observations expose the full reward candidate list, the prepared enemy skill list, status removal counters, and Gate order. Policies that receive those observations therefore have access to more information than policies restricted to what the original screen shows.

## 2. Skill setup and mana verification

The original tracks four distinct collections. It records learned skills in `playerskill_dex`, equipped skills in `current_skills`, uses per slot in `memorized_skills`, and the allocation saved on departure in `saved_skills`.

| Operation | State changes verified in the original |
|---|---|
| Learn a skill | Adds the final selection from the post-victory candidates to the learned skill list. Equipping the skill and allocating uses are separate operations. |
| Increase uses by 1 | When current mana covers 1 use, `memorize_skill` increases the count by 1 and recalculates mana. |
| Decrease uses by 1 | `forget_skill` reduces a positive use count by 1, down to a minimum of 0. The skill remains learned and equipped. |
| Use a skill in combat | Consumes 1 use when the action proceeds to effect resolution, including on a miss. Sacrifice and the player's Mimic copy action are handled separately. Casting leaves the original's stored `current_mana` value unchanged. |
| Replace a skill | The old skill remains learned, and a new skill takes its equipped slot. The original removes the old allocation, sets the new skill's uses to 0, and recalculates the total cost. The old allocation stays discarded if the player re-equips that skill. |
| Reduce mana cost | A skill is ineligible if its current cost is already below its starting cost or its starting cost was 1 or less. The one-time reduction costs 1000 gold and lowers the cost per use by 1. Allocated uses stay the same, and the mana balance is recalculated. |
| Increase maximum mana | A mana potion adds 10 to the maximum. The original also adds 10 to the stored current balance. The simulator displays the balance recalculated from the maximum. |
| Enter and leave the shop | On departure, the game saves a copy of the allocation and uses it again at the next shop. Learned skills persist across shop visits. |

The audit directly reviewed `script.rpy:362–364,637–652,1546–1580,1686–1742,1887–1898,4136–4161,4276–4286,4318–4347` and executed the original setup and economy functions in the private reference environment. After recalculating mana, it checked allocations, equipped skills, gold, mana, and stats across 14 consecutive operations. At the start with seed 24, the original's **stored balance was 170** and the simulator's **calculated balance was 0**. Both balances were 0 immediately after the original recalculated mana on entry to skill setup. A comparison of the stored field and a comparison of normal setup behavior measure different things.

## 3. Differences observed in the audit

F-01 and F-02 describe behavior at the audit baseline commit. See §5 for the final status.

### Doppelganger Imitation and reward timing

Doppelganger's only base skill is `imitation`. The comparison assumes the player has yet to learn `imitation`.

| When Doppelganger is defeated | Original | Simulator |
|---|---|---|
| The player goes first and defeats it before it casts Imitation | `imitation` is available as a reward. | Matches. A separate differential test found that the reward candidates and RNG state both matched. |
| Doppelganger casts Imitation first and is defeated **on the same turn** | `imitation` is available as a reward. The copy list exists, and that turn's reward list still uses the one base skill. | No skill reward. The copy list immediately becomes the list used for rewards. |
| Doppelganger is defeated **on the next turn** with Imitation still active | The copied skills become the candidates at the beginning of the turn. If the player has already learned all of them, there is no skill reward. | No skill reward under the same conditions. |
| Doppelganger first casts a copied Recovery on a turn when Imitation is active, then is defeated | Uses that turn's copied skill list for rewards. The tested case had no skill reward. | Immediately restores the base skill list and offers `imitation` as a candidate. |

| Category | ID | Observation and evidence |
|---|---|---|
| Implementation bug | F-01 | **Enemy KO immediately after applying Imitation.** The original builds a separate list of copied candidates in `copy_from_player` and retains `available_skills` for the current turn. The simulator immediately changes `available_skills` as well. In a controlled case where Doppelganger on Mansion floor 17 cast Imitation first and was defeated that turn, the original offered `["imitation"]` and the simulator offered `[]`. Their RNG states also diverged at the reward prompt. |
| Implementation bug | F-02 | **Enemy KO immediately after clearing Imitation.** The original retains that turn's copied skill list for the current reward. The simulator restores `available_skills` from the recovery snapshot as well. In a controlled case where Doppelganger first cast a copied Recovery and was defeated on the same turn, the original offered `[]` and the simulator offered `["imitation"]`. The main references for both cases are `script.rpy:411–439,522–557,2339–2346,4013–4017`, `_imitate` and `cure_status_effects` in `simulator/skills.py`, and `prepare` in `rewards.py`. |
| Missing functionality | F-03 | The original calls save for slot `1-1`, offers Load on the start menu, and attempts to delete the save at the ending label. The simulator has no save, restore, delete, or overwrite confirmation. It also has no save side effects on entering the shop or clicking Reroll. |
| Interface difference | F-04 | The simulator combines slot and inventory selection into a single replacement action, so there is no intermediate Back action. It exposes no standalone actions for no-op clicks, Lock, tooltips, or menu screens. `forget` corresponds to the original function name and means `deallocate one use`. Neither version offers a permanent skill removal action. |
| Interface difference | F-05 | The simulator derives observed mana from the current allocation. The original exposes a cached field. Seed 24 reproduced the initial balances of 170 and 0. The tested operations matched after entering skill setup. Explicit seeding and structured internal state are also separate programmatic interfaces. |
| Original bug | F-06 | Compiling `reroll_stats` in isolation raises `SyntaxError` because it references `player_gold` before the global declaration. The simulator implements the calculation specified in the source body. The audit left the original intact and excluded this branch from the matching cases. |
| Original bug | F-07 | If the first hit of Hybrid Attack lands and the second misses, the original attempts to return an undefined `hits`. Four cases with extreme accuracy values raised `UnboundLocalError`. The simulator retains the first hit. Whether those values can occur during ordinary play in the original remains unverified. |
| Original bug | F-08 | Clearing all statuses just after player Imitation raised `UnboundLocalError` on a local variable reference. Imitating Ella raised `KeyError` on a lookup in the fixed registry. The simulator continues using snapshot restoration and Ella's current skill list. Repeated Imitation also differs in how it handles duplicate backups and marker removal. The audit covered only some combinations in this repeated-use branch. |
| Original bug | F-09 | The original can skip the next status when it removes a status during iteration, overrides the floor 35 Ella choice in a later branch, and compares against `bonespear` when replacing a higher-tier enemy skill. The simulator preserved those behaviors and their RNG consumption in the cases audited. |
| Unverified | F-10 | Save and load behavior and engine RNG restoration, parity between the Steam build and the backed-up commit, the effect of localized string comparisons in the original, and whether the errors occur in an installed build remain unverified. The audit identified a general-purpose file menu and two quick menu declarations in `screens.rpy`. It did not run the engine to determine which menu is active or which normal key paths reach it. `options.rpy` disables rollback and quicksave, and clears game-menu key bindings. |

## 4. Private reference harness and verification results

The audit used differential tests in a separate private local workspace. The harness reads the pinned source files at runtime and **executes the original `Char` methods and the Python operations and conditions in selected Ren’Py labels**. The label translator accepts a limited grammar and fails on unsupported statements. Stubs handle screens, audio, dialogue, achievements, and saves. The reward screen passes through the selected value and reads the Gold increment from the original screen. The Ella menu executes the selected original branch. After reproducing the original compilation error, the harness excluded only the Reroll method from the in-memory compilation of the reference class so that it could test the other methods. The simulator and reference harness used separate objects and independently seeded RNGs with the same seed. The audit checked both versions' state and compared their RNG states wherever possible.

| Area | Coverage and results |
|---|---|
| Skills and combat | Checked the names and value tuples for **88** skills in the original registry. The **2,646** full-turn cases covered both casters, ordinary combat, low HP, high evasion, substitutes, overlapping statuses, and Hybrid edge cases. In **2,642** cases, the versions agreed on both characters' and substitutes' HP, compared stats, statuses, action choice, uses, gold, and RNG state. The remaining **4** cases reproduced the original Hybrid exception. F-01 and F-02 record the reward list differences observed at the unfixed baseline. |
| Status effects | All **78** cases matched across 11 individual statuses and lists of overlapping statuses in forward and reverse order, varying attack or Recovery and the seed. The audit compared blocked actions, HP before and after, up to 7 status removal ticks, status order, and RNG state. The immediate Imitation removal exception was reproduced separately. |
| Enemy AI | All **2,200** cases across 55 variants × 8 floor conditions × 5 status and stat conditions matched on the chosen skill, candidate list, and RNG state. |
| Rewards and shops | Compared progression while cycling through all five reward choices, and **14** shop operations after recalculating mana. The 2 Doppelganger reward discrepancies and the Reroll reference compilation failure were reproduced separately. |
| Floor progression | Forcing enemy defeats and advancing in both versions produced **101** encounters on route 1 and **100** on route 2. Floor, area, enemy ID and stats, Ella skills, learned skills, gold, and RNG state matched. The forced results let this test check routes and ending conditions. Performance in actual play falls outside its scope. |
| Defeat and clear conditions | Verified the ending branch and reward skip after a floor 100 victory on both routes. The 2 cases, a player-only KO and a simultaneous KO, both resulted in game over in the original and simulator. |

**Using the full Ren’Py engine as a reference.** A separate private checkout with the same Ren’Py SDK, game build, and resources could use test-only screen choices and seed injection to compare label transitions, screen behavior, and save and restore. The audit found no Ren’Py executable in the current environment and did not run the full project or a distribution build. These results cover the CPython adapter used in the audit. The original files and test harness stayed outside the public repository.

## 5. Final Phase 2 fixes and verification

- **F-01 and F-02 fixed.** Rewards use the skill list the enemy AI builds when it starts choosing an action for the turn. Applying or clearing Imitation affects the copied slots and character stats while retaining the current turn's list for rewards. On the next turn, the enemy AI selects a new list based on the current status.
- **F-04 labels clarified.** The CLI displays `Increase skill uses` and `Decrease skill uses`. The internal actions remain `memorize` and `forget`. Replacement still combines the two UI steps, and the simulator continues to omit no-op clicks.
- **Saving and mana observation (F-03 and F-05).** Save and resume are absent. Observed current mana is calculated from the live allocation. The [observation space documentation](observation_space.md) categorizes every observation key.
- **Original edge cases (F-06–F-09).** Reroll runs the calculation described in the source. A missed second Hybrid hit retains the first hit. Rogue Imitation restores from a snapshot and can use Ella's current skills. The status list iteration skip, the overridden random choice for floor 35 Ella, and the `bonespear` comparison persist. [Known differences and verification limits](known_issues.md)

The two regression tests confirmed that the pre-fix candidate lists were reversed in the two cases. All 58 tests passed after the fix. The private differential suite was rerun and found that **reward candidates and RNG states matched** for all three timings tested. These were a first-strike defeat before Imitation, a defeat on the same turn Imitation was applied, and a defeat on the same turn a copied Recovery was used. The previous comparison results held for 2,642 combat cases, 78 status cases, 2,200 enemy AI cases, 201 progression encounters, 14 shop operations, and 2 defeat cases. The 4 original Hybrid exceptions and the Reroll and Imitation reference exceptions still reproduced. Runtime parity with the Steam release remains unverified.
