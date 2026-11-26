"""Load and index the simulator's static game records."""
import json
from pathlib import Path

DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "data"


class GameData:
    def __init__(self, directory=DATA_DIRECTORY):
        directory = Path(directory)

        def read(name):
            return json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))

        self.skills = {entry["name"]: entry for entry in read("skills")}
        self.monsters = {entry["id"]: entry for entry in read("monsters")}
        self.areas = read("areas")
        self.regions = {entry["id"]: entry for entry in self.areas["regions"]}
        self.config = read("game_config")
