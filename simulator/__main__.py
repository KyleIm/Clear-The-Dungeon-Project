"""Terminal adapter for the simulator. Run with python -m simulator."""
import argparse

from .game import Game


def show_state(state):
    player, monster = state["player"], state["monster"]
    print(f"\nFloor {state['floor']} | {state['area']} | {state['phase']} | Gold {state['gold']}")
    print(f"Rogue HP {player['hp']}/{player['max_hp']} | ATK {player['attack']} | "
          f"DEF {player['defense']} | SPD {player['speed']} | ACC {player['ACC']} | "
          f"EVA {player['EVA']} | DR {player['DR']}")
    status = ", ".join(effect["Name"] for effect in player["status_conditions"])
    if status:
        print(f"Rogue conditions: {status}")
    if state["phase"] == "combat":
        print(f"{monster['name']} HP {monster['hp']}/{monster['max_hp']} | ATK {monster['attack']} | "
              f"DEF {monster['defense']} | SPD {monster['speed']} | ACC {monster['ACC']} | "
              f"EVA {monster['EVA']} | DR {monster['DR']}")
        status = ", ".join(effect["Name"] for effect in monster["status_conditions"])
        if status:
            print(f"Opponent conditions: {status}")
    elif state["phase"] == "reward":
        print(f"Skill offer: {state['reward_skill'] or 'none'}")
    elif state["phase"] == "shop":
        print(f"Shop menu: {state['shop_menu']} | Remaining mana: {player['current_mana']} | "
              f"Unallocated stat points: {player['available_points']}")
        print(", ".join(f"{name} ({uses})" for name, uses in player["memorized_skills"].items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int)
    options = parser.parse_args()
    game = Game()
    state = game.reset(options.seed)
    while not game.is_terminal():
        show_state(state)
        actions = game.get_legal_actions()
        for index, action in enumerate(actions, start=1):
            label = " ".join(map(str, action))
            if action[0] == "skill" and action[1] in state["player"]["memorized_skills"]:
                label += f" ({state['player']['memorized_skills'][action[1]]} uses)"
            print(f"{index}. {label}")
        try:
            choice = input("Action number, or q to quit: ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if choice.lower() == "q":
            return
        try:
            index = int(choice) - 1
            if not 0 <= index < len(actions):
                raise ValueError
        except ValueError:
            print("Enter a listed action number.")
            continue
        state = game.step(actions[index])
    show_state(state)
    print("Cleared!" if game.is_cleared() else "Game over.")


if __name__ == "__main__":
    main()
