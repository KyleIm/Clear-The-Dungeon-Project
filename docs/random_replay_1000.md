# Million-game baseline replay check

The current simulator reproduced the furthest floor, action count, outcome and termination detail for all 1,000 selected episodes. The sample contained 913 defeats and 87 action-limit timeouts. All 550 episodes observed entering a shop matched the historical record on all four fields.

Each replay used the original stored game and policy seeds and the baseline runner's action and wall-clock limits. The source SQLite database was opened read-only. The sampling PRNG used seed 314159 and sampled from each stratum's ascending game IDs in the order below. Shop entry was verified from simulator phase transitions during replay.

| Historical stratum | Selected | Replayed with a shop visit |
|---|---:|---:|
| Floor 11 or higher | 550 | 550 |
| Floor 10 | 100 | 0 |
| Action-limit timeout | 87 | 0 |
| Defeat below floor 10 | 263 | 0 |

The sample deliberately overrepresents shop visits and timeouts. Its outcome proportions describe the sample. [Per-game seeds and comparisons](random_replay_1000_verification.json) · [Original baseline report](baseline_random_1m_results.md)
