"""Run-local character state shared by combat and the economy."""
from dataclasses import dataclass, field


@dataclass
class Character:
    key: str
    name: str
    base_type: str
    base_attack: int
    base_defense: int
    base_hp: int
    base_speed: int
    base_EVA: int = 0
    base_ACC: int = 0
    base_DR: int = 0
    monster_id: int = 0
    type: str = field(init=False)
    attack: int = field(init=False)
    defense: int = field(init=False)
    hp: int = field(init=False)
    speed: int = field(init=False)
    EVA: int = field(init=False)
    ACC: int = field(init=False)
    DR: int = field(init=False)
    status_conditions: list[dict] = field(default_factory=list)
    current_skills: list[str] = field(default_factory=list)
    memorized_skills: dict[str, int] = field(default_factory=dict)
    saved_skills: dict[str, int] = field(default_factory=dict)
    learned_skills: list[str] = field(default_factory=list)
    available_skills: list[str] = field(default_factory=list)
    skill_costs: dict[str, int] = field(default_factory=dict)
    max_mana: int = 170
    available_points: int = 0
    purchases: set[str] = field(default_factory=set)
    selected_action: str | None = None
    ram_action: str | None = None
    ram_damage: int = 0
    priority_level: int = 0
    mm_button: int = 0
    substitute_hit_flag: int = 0
    imitation_backup: dict | None = None

    def __post_init__(self):
        self.type = self.base_type
        self.attack = self.base_attack
        self.defense = self.base_defense
        self.hp = self.max_hp
        self.speed = self.base_speed
        self.EVA = self.base_EVA
        self.ACC = self.base_ACC
        self.DR = self.base_DR

    @classmethod
    def from_record(cls, record):
        base, extra = record["base_stats"], record["additional_stats"]
        character = cls(
            key=record["key"], name=record["name"], base_type=record["type"],
            base_attack=base["attack"], base_defense=base["defense"],
            base_hp=base["hp"], base_speed=base["speed"],
            base_EVA=extra["evade"], base_ACC=extra["accuracy"],
            base_DR=extra["damage_reduction"], monster_id=record.get("id", 0),
        )
        character.available_skills = list(record.get("skills", ()))
        return character

    @property
    def max_hp(self):
        return self.base_hp * 2

    @property
    def current_mana(self):
        return self.max_mana - sum(
            self.skill_costs[skill] * count for skill, count in self.memorized_skills.items()
        )

    def has_status_effect(self, name):
        return any(effect["Name"] == name for effect in self.status_conditions)

    def apply_status_effect(self, name):
        self.status_conditions.append({"Name": name, "chance_to_cure": 0})

    def wake_up(self):
        for effect in self.status_conditions:
            if effect["Name"] == "sleep":
                self.status_conditions.remove(effect)
