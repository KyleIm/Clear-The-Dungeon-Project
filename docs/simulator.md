# Python simulator

The simulator reads the Phase 1 data and runs the game as a sequence of decisions. It uses Python 3.10 or newer and the standard library. The reference revision is recorded in [game_config.json](../data/game_config.json). The [mechanics reference](game_mechanics.md) describes the source rules and the [issue list](known_issues.md) records source anomalies.

## Terminal play

From the repository root, run:

```console
python -m simulator --seed 42
```

Choose a numbered action. The terminal adapter shows different choices at combat, reward, story event, and shop boundaries. Enter `q` to exit.

## Headless play

```python
import random

from simulator.game import Game
from simulator.headless import play_headless

policy_rng = random.Random(7)
final_state = play_headless(
    Game(),
    lambda state, actions: policy_rng.choice(actions),
    seed=42,
)
```

The headless runner performs no console I/O. Its step limit guards against policies that remain in a shop indefinitely. It raises `TimeoutError` at the limit.

A caller can drive the engine directly:

```python
game = Game()
state = game.reset(seed=42)
while not game.is_terminal():
    choices = game.get_legal_actions()
    action = ("leave_shop",) if ("leave_shop",) in choices else choices[0]
    state = game.step(action)
cleared = game.is_cleared()
```

`get_state()` returns a detached, JSON-serializable observation. Use `get_legal_actions()` to enumerate the current choices. Each action is a tuple beginning with a command name, such as `("skill", "wolf fang")`, `("reward", "Gold")`, `("ella", "help")`, or `("leave_shop",)`. Equivalent JSON lists work with `step`. An illegal action raises `ValueError` before changing state or consuming random draws. Randomness in the policy is independent of engine randomness.

## Implementation and verification

- `game.py` coordinates combat, rewards, shops, Ella choices, and terminal transitions.
- `data_loader.py` and `character.py` provide the static data index and run-local state.
- `skills.py` and `combat.py` own numerical effects and turn resolution.
- `enemy_ai.py` and `dungeon.py` own encounter selection and enemy decisions.
- `rewards.py` and `shop.py` own player choices after combat. `headless.py` and `__main__.py` are separate adapters so the engine need not know about a policy loop or the terminal.

```console
python -m unittest discover -s tests -v
python tests/validate_game_data.py
```

The tests cover execution of all 88 skill entries, targeted effect and subsystem cases, seed replay, random play, and both 1–100 progression routes. The full progression tests force encounter wins to exercise later floors regardless of policy strength. Local comparisons against extracted source routines also covered ordinary skill resolution and enemy action selection. A separate 256-episode uniform-random smoke run reached floor 10 and recorded zero clears. The Steam build has yet to be compared in runtime tests.

## Boundary and known differences

- The simulator exposes decision states and typed actions in place of the Ren'Py screens. It skips visual, audio, dialogue, achievement, and save-slot operations. Saved-game resume/rollback and Steam input automation are outside Phase 2.
- Explicit seeds are reproducible within the simulator. A separate Python random stream is derived from the seed for shop rerolls. The original uses both `renpy.random` and Python `random`; seed equivalence, RNG restoration, and rollback behavior in the game engine are unverified.
- The observation's mana is computed from current skill uses. The original stores a balance and refreshes it at allocation points. The simulator enforces the allocation budget against the current shop slots and uses.
- The action list omits UI operations whose source result is a no-op, such as adding a use without enough mana. It retains valid moves that consume a resource even if their effect fails, such as a key purchase with no eligible replacement.
- The simulator models the arithmetic in the reroll branch despite the source's global-declaration hazard. It uses explicit run-local Imitation backup state to make recovery deterministic. See [known issues](known_issues.md) for code-level discrepancies and verification limits.
