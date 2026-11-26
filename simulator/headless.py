"""Run policy-driven games without console input or output."""


def play_headless(game, policy, *, seed=None, max_steps=100_000):
    """Call policy(state, legal_actions) until terminal and return the final state.

    Policy randomness is managed by the caller and is independent of game RNG.
    """
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    state = game.reset(seed)
    for _ in range(max_steps):
        action = policy(state, game.get_legal_actions())
        state = game.step(action)
        if game.is_terminal():
            return state
    raise TimeoutError(f"Policy did not finish within {max_steps} actions")
