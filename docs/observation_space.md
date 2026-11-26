# Simulator observation space

This is an inventory of the current `Game.get_state()` snapshot, classified against the Dungeon & Rogue 1.00 Ren'Py source. It describes information availability, not a training interface. No feature vector, observation filter, or model is implemented.

- **Direct** means the source displays the field or its equivalent to the player on at least one relevant screen or prompt. Visibility can depend on the phase, hover state, or presence of a substitute.
- **History** means a player can record or reconstruct the field from prior choices and visible events, sometimes with known game rules. A value may be less certain without a complete history.
- **Internal** means the exact field contains information the source UI does not reliably reveal. A documented rule, a fixed default, or a partial observation does not make its full run-time value directly observable.

The reference is the pinned source revision in [game_config.json](../data/game_config.json). The classification comes from the source screens and labels. Visibility in the installed Steam build, localization, and accessibility behavior have not been validated.

## Run-level fields

| Field | Category | Source availability |
|---|---|---|
| `phase` | Direct | Combat, reward prompt, encounter choice, shop, ending, and defeat appear as different screens or prompts. The Python phase IDs are labels for these states. |
| `floor`, `area`, `gold` | Direct | The HUD shows area, floor, and gold. The shop also shows gold in its stat view. |
| `shop_menu` | Direct | The active shop screen identifies skill allocation, cost reduction, redistribution, or merchandise. The stored default outside shop is a simulator label. |
| `reward_skill` | Direct | The source announces the offered skill before the reward menu and announces when all eligible skills are already known. |
| `reroll_cost` | Direct | The merchandise screen prints the current price. |
| `area_number` | History | The counter starts at one and increments when leaving a shop. It differs from counting only visited regular areas, especially on the floor-99 help path. |
| `turn`, `wins` | History | Count completed combat decisions in the current encounter and defeated encounters across the run. The two fights on floor 99 can count separately. |
| `gate_queue` | History | The remaining order follows the sequence in which the stage bosses were encountered, minus the Gate fights already started. Recording those names and resolving their IDs reproduces the queue. |
| `ella_path`, `ella_encounter` | History | Track the floor-35 choice and the scripted floor-99 branch. The source presents their consequences as encounters and choices. |
| `remaining_areas` | Internal | Set membership can be reconstructed from area visits. The exact shuffled list order is not displayed. |
| `reward_candidates` | Internal | The source shows the drawn offer. Prior casts and learned skills can suggest candidates, but the full eligible list is not shown. |
| `ella_skills` | Internal | Casts can reveal individual entries. The full generated and potentially upgraded list is not shown. |
| `player`, `monster`, `player_substitute`, `monster_substitute` | See below | These are containers with a mix of direct, historical, and internal fields. |

Evidence in the original files includes `custom_screen.rpy:46–60,360–370,464–476` and `script.rpy:1923–1928,3761–3870,4013–4067,4080–4187,4343–4369`.

## Character fields

The four character snapshots expose the same model fields. The player also has `current_mana`, and each snapshot adds `max_hp`. A field's visibility differs by role because only the player uses a skill book and the substitute has a limited HP display. In the table, **P** means player, **M** means main enemy, and **S** means either substitute.

| Field | P | M | S | Source availability or limit |
|---|---|---|---|---|
| `name`, `hp` | Direct | Direct | Direct | The HP bar shows name and current HP as text. A substitute's bar replaces the corresponding body bar while active. The hidden body's HP then requires prior observations. |
| `type` | Direct | Direct | Internal | Current types for bodies are on the combat information panels. The substitute bar has no type label. |
| `attack`, `defense`, `speed` | Direct | Direct | Internal | The combat information panels show current values for the bodies. |
| `EVA`, `ACC`, `DR` | Direct | Direct | Internal | The player stat panel shows values. The enemy stat display prints each nonzero value and omits zero-valued entries. |
| `status_conditions[].Name` | Direct | Direct | Internal | The source displays icons for active body statuses, not the ordered records. Presence is directly visible when the body bar is shown. |
| `status_conditions[].chance_to_cure` and exact list order or multiplicity | Internal | Internal | Internal | The icons do not reveal counters, order, or duplicates. Application history may suggest a counter but can miss list-removal effects. |
| `current_skills` | Direct | Internal | Internal | The player's ordered equipment and locked slots are visible. The enemy's full copied slot list is not shown. It can sometimes be inferred from the player's slots at the time of copying. |
| `memorized_skills` | Direct | Internal | Internal | Uses for equipped player skills appear in combat and the skill book. The other roles' corresponding engine fields do not have a visible use counter. |
| `skill_costs` | Direct for learned skills, internal for the rest | Internal | Internal | The cost screen shows the current cost for each learned skill. The full registry-shaped map includes unlearned entries. |
| `max_mana` | Direct | Internal | Internal | The player's skill-book mana bar shows the maximum. |
| `current_mana` | Direct in skill management, History outside it | — | — | The source refreshes and shows the allocation balance on opening skill management. Elsewhere the simulator computes it from live uses. The hidden Ren'Py cached value can differ. |
| `available_points` | Direct | Internal | Internal | Only the player redistribution screen shows unallocated points. |
| `base_attack`, `base_defense`, `base_hp`, `base_speed` | Direct on redistribution screen | History where the opening value is identifiable | Internal | The player sees the four base values in redistribution. Enemy base values may be inferred before modifiers, from a known starting HP, or from past observations of that variant. Current enemy stats generally cannot reveal base values after arbitrary changes. |
| `base_type`, `base_EVA`, `base_ACC`, `base_DR` | History | Internal | Internal | A player can track baseline type and additional stats using story choices, purchases, permanent rewards, and reset observations. Enemy base values are separate from current displayed values and may include temporary modifiers or per-encounter changes. |
| `max_hp` | History | History | Internal | Body bar ranges use twice base HP. Fully healed HP and visible base HP can give the numerical cap; the bar does not print it separately. The substitute's source bar uses 20, whereas this simulator field is computed from its base HP and equals 0. |
| `learned_skills` | History | Internal | Internal | The player can reconstruct the learned set from rewards and the inventory/cost screens. Exact snapshot order and lock placeholders are bookkeeping. |
| `saved_skills` | History | Internal | Internal | The player can record the slot allocation when leaving a shop. The underlying snapshot is not shown separately. |
| `purchases` | History | Internal | Internal | Record the successful key and mana-potion purchases; available items also reveal some purchase flags. The set is not a directly displayed purchase ledger. |
| `key`, `monster_id` | Internal | Internal | Internal | Engine and registry identifiers. A displayed name can identify many enemies, but several variants share the name Ella. |
| `ram_action` | History | Internal | Internal | The player can record the action they selected. Some enemy actions are announced, but a selected move may never be shown if the enemy dies before acting. |
| `selected_action` | History | Internal | Internal | Player selections and Mimic casts can be tracked from choices and announcements. The exact enemy field can be hidden and may change after Mimic. |
| `available_skills` | Internal | Internal | Internal | The exact current enemy turn list is separate from the visible casting result and the copied slots. For the other roles it is bookkeeping. |
| `ram_damage`, `priority_level`, `mm_button`, `substitute_hit_flag` | Internal | Internal | Internal | These are execution flags or temporary records. Visible hits, action order, and substitute displays can reveal effects of the flags. |
| `imitation_backup` | Internal | Internal | Internal | The snapshot is either null or a restoration object containing `base_attack`, `attack`, `base_defense`, `defense`, `base_ACC`, `ACC`, `base_EVA`, `EVA`, `base_DR`, `DR`, `type`, `current_skills`, and `memorized_skills`. Its contents are not exposed as a backup in the game UI even if some pre-copy values were previously seen. |

Character screen evidence includes `custom_screen.rpy:50–247,250–358,406–460`. Announcements, Imitation, and turn state are in `script.rpy:411–471,480–635,1546–1580,2334–2525,2527–3362,4343–4347`. Complete reward candidates, exact region-list ordering, and hidden enemy actions are simulator information a screen-limited policy would not receive directly.
