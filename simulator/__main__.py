"""Terminal adapter for the simulator. Run with python -m simulator."""
import argparse

from .game import Game

ACTION_LABELS = {"memorize": "Increase skill uses", "forget": "Decrease skill uses"}


def _conditions(actor):
    return ", ".join(dict.fromkeys(effect["Name"] for effect in actor["status_conditions"]))


def show_state(state, skills):
    player, monster = state["player"], state["monster"]
    combat = state["phase"] == "combat"
    player_type = f" | Type {player['type']}" if combat else ""
    print(f"\nFloor {state['floor']} | {state['area']} | {state['phase']} | Gold {state['gold']}")
    print(f"Rogue HP {player['hp']}/{player['max_hp']}{player_type} | ATK {player['attack']} | "
          f"DEF {player['defense']} | SPD {player['speed']} | ACC {player['ACC']} | "
          f"EVA {player['EVA']} | DR {player['DR']}")
    status = _conditions(player)
    if combat or status:
        print(f"Rogue conditions: {status or 'none'}")
    if combat:
        print(f"{monster['name']} HP {monster['hp']}/{monster['max_hp']} | Type {monster['type']} | ATK {monster['attack']} | "
              f"DEF {monster['defense']} | SPD {monster['speed']} | ACC {monster['ACC']} | "
              f"EVA {monster['EVA']} | DR {monster['DR']}")
        print(f"Opponent conditions: {_conditions(monster) or 'none'}")
        depleted = [f"{name} [{skills[name]['type']}]" for name in player["current_skills"]
                    if name != "lock" and player["memorized_skills"][name] == 0]
        if depleted:
            print(f"Depleted skills: {', '.join(depleted)}")
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
        show_state(state, game.data.skills)
        actions = game.get_legal_actions()
        for index, action in enumerate(actions, start=1):
            label = " ".join((ACTION_LABELS.get(action[0], action[0]), *map(str, action[1:])))
            if action[0] == "skill":
                name = action[1]
                label += f" [{game.data.skills[name]['type']}]"
                if name in state["player"]["memorized_skills"]:
                    label += f" ({state['player']['memorized_skills'][name]} uses)"
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
    show_state(state, game.data.skills)
    print("Cleared!" if game.is_cleared() else "Game over.")


if __name__ == "__main__":
    main()
