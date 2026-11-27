# Combat display audit

The terminal combat screen now shows both combatants' current types and active condition names. Each listed combat skill has its type from the game data, including the Sacrifice fallback. Depleted equipped skills retain their name and type in a separate line. Action numbers and availability still come from the simulator.

The following information is present in the pinned Ren’Py source and remains missing from the terminal combat screen.

| Information | Original display | Terminal gap |
|---|---|---|
| Substitute identity and current HP | Each side's HP bar switches to its substitute while that substitute is active. `custom_screen.rpy:68–101,141–174` | The terminal shows body HP and gives no separate indication that a substitute is absorbing attacks. |
| Full slot layout and visible lock markers | The ordinary skill selection screen shows equipped slots in order, usage counts, depleted indicators and locked slots. When all available charges are gone, it shows the Sacrifice prompt. `custom_screen.rpy:250–274` | Action rows and the depleted summary show skills and charges. Locked slots and the original equipped positions are absent. |
| Skill descriptions | Hovering over usable combat skills, including Sacrifice, displays their description from the source skill registry. Which details are explained varies by skill. `custom_screen.rpy:250–274` | The terminal shows skill name, type and applicable uses. It has no effect, accuracy or targeting tooltip. |
| Status and extra-stat explanations | Condition icons show hover descriptions on both sides. Visible enemy EVA, ACC and DR icons have their own tooltips. `custom_screen.rpy:104–138,177–229` | The terminal gives the condition names and stat values without these explanations. |
| Resolved actions and effects | The battle labels announce casts, failed actions and outcomes such as damage, healing, recoil, applied or removed conditions, and copied moves. `script.rpy:2450–2524,2527–2972,2974–3362` | The terminal renders the next snapshot without a turn transcript. The player can compare successive snapshots to infer some outcomes. |

The original stat panels show the current types and the main combat stats. The terminal already shows the numeric attack, defense, speed, accuracy, evasion and damage reduction for both sides. Exact cure counters and unrevealed enemy selections are internal and do not belong in the list of missing player-visible information. This audit checked the backed-up source. The installed Steam build was outside its scope. [Observation inventory](observation_space.md)
