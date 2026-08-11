import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

@dataclass
class MCWorldState:
    health: float = 20.0
    max_health: float = 20.0
    hunger: float = 20.0
    saturation: float = 5.0
    pos_x: float = 0.0
    pos_y: float = 64.0
    pos_z: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    dimension: str = "minecraft:overworld"
    biome: str = "minecraft:plains"
    current_goal: str = "NONE"
    in_combat: bool = False
    gamemode: str = "SURVIVAL"
    experience: int = 0
    time_of_day: int = 0
    is_raining: bool = False
    is_night: bool = False
    armor_protection: float = 0.0
    is_sprinting: bool = False
    is_sneaking: bool = False
    is_swimming: bool = False
    on_ground: bool = True
    is_in_water: bool = False
    is_in_lava: bool = False
    nearby_hostiles: int = 0
    nearby_items: int = 0
    inventory: List[Dict] = field(default_factory=list)
    hotbar: List[Dict] = field(default_factory=list)
    selected_slot: int = 0
    block_looking_at: Optional[str] = None

    @classmethod
    def from_json(cls, json_str: str) -> 'MCWorldState':
        data = json.loads(json_str)
        state = cls()
        field_map = {
            'health': 'health', 'maxHealth': 'max_health', 'hunger': 'hunger',
            'saturation': 'saturation', 'posX': 'pos_x', 'posY': 'pos_y',
            'posZ': 'pos_z', 'yaw': 'yaw', 'pitch': 'pitch',
            'dimension': 'dimension', 'biome': 'biome',
            'currentGoal': 'current_goal', 'inCombat': 'in_combat',
            'gamemode': 'gamemode', 'experience': 'experience',
            'timeOfDay': 'time_of_day', 'isRaining': 'is_raining',
            'isNight': 'is_night', 'armorProtection': 'armor_protection',
            'isSprinting': 'is_sprinting', 'isSneaking': 'is_sneaking',
            'isSwimming': 'is_swimming', 'onGround': 'on_ground',
            'isInWater': 'is_in_water', 'isInLava': 'is_in_lava',
            'nearbyHostiles': 'nearby_hostiles', 'nearbyItems': 'nearby_items',
            'inventory': 'inventory', 'hotbar': 'hotbar',
            'selectedSlot': 'selected_slot', 'blockLookingAt': 'block_looking_at',
        }
        for key, value in data.items():
            mapped = field_map.get(key, key)
            if hasattr(state, mapped):
                setattr(state, mapped, value)
        return state

    def to_state_vector(self) -> np.ndarray:
        vec = np.zeros(141, dtype=np.float32)

        vec[0] = min(self.health / self.max_health, 1.0)
        vec[1] = min(self.hunger / 20.0, 1.0)
        vec[2] = (self.pos_x % 1000) / 1000.0
        vec[3] = self.pos_y / 256.0
        vec[4] = np.sin(np.radians(self.yaw))
        vec[5] = np.cos(np.radians(self.yaw))

        hostiles_normalized = min(self.nearby_hostiles / 10.0, 1.0)
        for i in range(5):
            offset = 6 + i * 4
            if i < self.nearby_hostiles:
                vec[offset] = 0.3
                vec[offset + 1] = 0.0
                vec[offset + 2] = 0.5
                vec[offset + 3] = 1.0

        for i in range(5):
            offset = 26 + i * 4
            vec[offset] = 0.0
            vec[offset + 1] = 0.0
            vec[offset + 2] = 1.0
            vec[offset + 3] = 0.0

        vec[46] = 0.0
        vec[47] = 0.0
        vec[48] = min(self.health / self.max_health, 1.0)
        vec[49] = min(self.time_of_day / 24000.0, 1.0)

        vec[119] = 1.0
        vec[120] = min(self.hunger / 20.0, 1.0)
        vec[121] = 0.5
        vec[122] = 0.0
        vec[123] = 0.0
        vec[124] = 0.0
        vec[125] = 0.0
        vec[126] = 0.0
        vec[127] = 0.0
        vec[128] = 0.0
        vec[129] = 1.0 if self.in_combat else 0.0
        vec[130] = 0.0
        vec[131] = min(self.nearby_items / 10.0, 1.0)
        vec[132] = 1.0 if self.is_raining else 0.0
        vec[133] = 1.0 if self.is_night else 0.0
        vec[134] = 0.0
        vec[135] = 1.0 if self.is_swimming else 0.0
        vec[136] = 0.0
        vec[137] = min(len(self.inventory) / 36.0, 1.0)
        vec[138] = 0.0
        vec[139] = 0.0
        vec[140] = 0.0

        for i in range(50):
            if i >= 141:
                break

        return vec


class MCPerception:
    def __init__(self, state_buffer_size: int = 50):
        from src.perception.time_buffer import TimeBuffer
        self.buffer = TimeBuffer({'buffer_size': state_buffer_size})
        self.state_vector_size = 141

    def process_state(self, world_state: MCWorldState) -> Dict:
        state_vector = world_state.to_state_vector()

        semantic_data = self._build_semantic_data(world_state)
        mapped_state = self._build_mapped_state(world_state)

        entry = {
            'frame_id': getattr(self, '_frame_counter', 0),
            'state_vector': state_vector,
            'semantic_data': semantic_data,
            'mapped_state': mapped_state,
            'notifications': [],
            'detection': self._build_detection(world_state),
            'ocr_texts': [],
        }
        setattr(self, '_frame_counter', entry['frame_id'] + 1)

        self.buffer.push(entry)
        return entry

    def get_mlp_input(self) -> np.ndarray:
        return self.buffer.sample_for_mlp()

    def get_decision_history(self) -> List[Dict]:
        return self.buffer.sample_for_decision()

    def _build_semantic_data(self, state: MCWorldState) -> Dict:
        hp_percent = int((state.health / state.max_health) * 100)
        hp_status = "健康" if hp_percent > 70 else ("受伤" if hp_percent > 30 else "残血")

        return {
            'self_hp': state.health,
            'self_hp_percent': hp_percent,
            'self_hp_status': hp_status,
            'self_hunger': int(state.hunger),
            'in_combat': state.in_combat,
            'nearby_hostiles': state.nearby_hostiles,
            'nearby_items': state.nearby_items,
            'dimension': state.dimension,
            'biome': state.biome,
            'time_of_day': state.time_of_day,
            'is_night': state.is_night,
            'current_goal': state.current_goal,
            'experience': state.experience,
            'armor': state.armor_protection,
            'pos_summary': f"({state.pos_x:.0f},{state.pos_y:.0f},{state.pos_z:.0f})",
            'inventory_count': len(state.inventory),
            'hotbar_count': len(state.hotbar),
            'block_looking_at': state.block_looking_at or "none",
        }

    def _build_mapped_state(self, state: MCWorldState) -> str:
        hp = state.health
        hp_pct = int(hp / state.max_health * 100)
        return (
            f"HP:{hp_pct}% H:{int(state.hunger)} "
            f"Dim:{state.dimension.split(':')[-1]} "
            f"Goal:{state.current_goal} "
            f"Hostiles:{state.nearby_hostiles} "
            f"Night:{1 if state.is_night else 0} "
            f"Fight:{1 if state.in_combat else 0}"
        )

    def _build_detection(self, state: MCWorldState) -> Dict:
        return {
            'red': 0,
            'green': 0,
            'enemy_count': state.nearby_hostiles,
            'ally_count': 0,
        }


def minecraft_state_to_mlp_input(state: MCWorldState) -> np.ndarray:
    """Convert MCWorldState directly to MLP input vector (456 dims)."""
    perception = MCPerception()
    entry = perception.process_state(state)
    return perception.get_mlp_input()
