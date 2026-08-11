import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MCActionSpace:
    MOUSE_X: int = 0
    MOUSE_Y: int = 1
    MOVE_X: int = 2
    MOVE_Y: int = 3
    LMB: int = 4
    RMB: int = 5
    MMB: int = 6
    KEY_W: int = 7
    KEY_A: int = 8
    KEY_S: int = 9
    KEY_D: int = 10
    KEY_SPACE: int = 11
    KEY_SHIFT: int = 12
    KEY_CTRL: int = 13
    KEY_TAB: int = 14
    KEY_ESC: int = 15
    KEY_1: int = 16
    KEY_2: int = 17
    KEY_3: int = 18
    KEY_4: int = 19
    KEY_5: int = 20
    KEY_6: int = 21
    KEY_7: int = 22
    KEY_8: int = 23
    KEY_9: int = 24
    KEY_0: int = 25
    KEY_Q: int = 26
    KEY_E: int = 27
    KEY_R: int = 28
    KEY_F: int = 29
    KEY_G: int = 30
    KEY_H: int = 31
    KEY_Z: int = 32
    KEY_X: int = 33
    KEY_C: int = 34
    KEY_V: int = 35
    KEY_B: int = 36
    KEY_N: int = 37
    KEY_T: int = 38
    KEY_M: int = 39

    ACTION_DIM: int = 40

    @staticmethod
    def to_minecraft_command(action_vector: np.ndarray) -> List[str]:
        """Convert 40-dim action vector to Minecraft bot commands."""
        commands = []

        mouse_x = (float(action_vector[0]) - 0.5) * 2.0
        mouse_y = (float(action_vector[1]) - 0.5) * 2.0
        move_x = (float(action_vector[2]) - 0.5) * 2.0
        move_y = (float(action_vector[3]) - 0.5) * 2.0

        if abs(mouse_x) > 0.1 or abs(mouse_y) > 0.1:
            yaw_change = mouse_x * 10.0
            pitch_change = mouse_y * 5.0
            commands.append(f"LOOK:{yaw_change:.1f}:{pitch_change:.1f}")

        if abs(move_x) > 0.1 or abs(move_y) > 0.1:
            commands.append(f"MOVE:{move_x:.2f}:{move_y:.2f}")

        if float(action_vector[MCActionSpace.KEY_Q]) > 0.5:
            commands.append("DROP")

        key_map = {
            MCActionSpace.KEY_W: 'W',
            MCActionSpace.KEY_A: 'A',
            MCActionSpace.KEY_S: 'S',
            MCActionSpace.KEY_D: 'D',
            MCActionSpace.KEY_SPACE: 'SPACE',
            MCActionSpace.KEY_SHIFT: 'SHIFT',
            MCActionSpace.KEY_CTRL: 'CTRL',
            MCActionSpace.KEY_E: 'E',
            MCActionSpace.KEY_F: 'F',
            MCActionSpace.KEY_1: 'HOTBAR_1',
            MCActionSpace.KEY_2: 'HOTBAR_2',
            MCActionSpace.KEY_3: 'HOTBAR_3',
        }

        for idx, name in key_map.items():
            if float(action_vector[idx]) > 0.5:
                commands.append(f"PRESS:{name}")

        if float(action_vector[MCActionSpace.LMB]) > 0.5:
            commands.append("LMB")
        if float(action_vector[MCActionSpace.RMB]) > 0.5:
            commands.append("RMB")

        return commands


class MCPolicy(nn.Module):
    def __init__(self,
                 state_dim: int = 141,
                 time_points: int = 3,
                 embed_dim: int = 32,
                 hidden_sizes: List[int] = (512, 1024, 512),
                 output_dim: int = 40,
                 num_strategies: int = 16):
        super().__init__()

        self.state_dim = state_dim
        self.time_points = time_points
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        input_size = state_dim * time_points + embed_dim + 1

        self.strategy_embed = nn.Embedding(num_strategies, embed_dim)

        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_dim))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)

    def forward(self,
                state: torch.Tensor,
                strategy_idx: torch.Tensor,
                game_type: torch.Tensor) -> torch.Tensor:
        strategy_emb = self.strategy_embed(strategy_idx)
        batch_size = state.shape[0]
        flat_state = state.view(batch_size, self.state_dim * self.time_points)
        game_type = game_type.unsqueeze(-1).float() / 10.0
        x = torch.cat([flat_state, strategy_emb, game_type], dim=-1)
        return self.net(x)

    def get_action(self,
                   state_vector: np.ndarray,
                   strategy: str,
                   game_type_id: int = 5) -> np.ndarray:
        strategy_map = {
            'GATHER_WOOD': 0, 'MINE_STONE': 1, 'MINE_IRON': 2,
            'MINE_DIAMOND': 3, 'CRAFT_TOOLS': 4, 'CRAFT_ARMOR': 5,
            'FIND_FOOD': 6, 'BUILD_SHELTER': 7, 'EXPLORE': 8,
            'FIND_LAVA': 9, 'BUILD_PORTAL': 10, 'ENTER_NETHER': 11,
            'FIND_FORTRESS': 12, 'KILL_BLAZE': 13,
            'FIND_END_PORTAL': 14, 'KILL_ENDER_DRAGON': 15,
        }
        strategy_idx = strategy_map.get(strategy.upper(), 0)

        self.eval()
        with torch.no_grad():
            if state_vector.ndim == 1 and state_vector.shape[0] == self.state_dim:
                full_input = np.concatenate([
                    state_vector,
                    np.zeros(self.state_dim),
                    np.zeros(self.state_dim),
                    np.zeros(self.embed_dim),
                    np.array([float(game_type_id)]),
                ]).astype(np.float32)
            elif state_vector.ndim == 1 and state_vector.shape[0] >= self.state_dim * self.time_points:
                padding_needed = self.state_dim * self.time_points + self.embed_dim + 1 - len(state_vector)
                if padding_needed > 0:
                    state_vector = np.concatenate([state_vector, np.zeros(padding_needed)])
                full_input = state_vector[:self.state_dim * self.time_points + self.embed_dim + 1].astype(np.float32)
            else:
                full_input = state_vector.astype(np.float32)

            state_tensor = torch.from_numpy(full_input).float().unsqueeze(0)
            total_dims = self.state_dim * self.time_points
            batch_state = state_tensor[:, :total_dims].view(1, self.state_dim, self.time_points)
            batch_state = batch_state.permute(0, 2, 1).reshape(1, self.state_dim * self.time_points)
            strategy_tensor = torch.tensor([strategy_idx], dtype=torch.long)
            game_type_tensor = torch.tensor([game_type_id], dtype=torch.long)
            action = self.forward_native(batch_state, strategy_tensor, game_type_tensor)
            return action.squeeze(0).cpu().numpy()

    def forward_native(self,
                       flat_state: torch.Tensor,
                       strategy_idx: torch.Tensor,
                       game_type: torch.Tensor) -> torch.Tensor:
        strategy_emb = self.strategy_embed(strategy_idx)
        game_type_scalar = game_type.unsqueeze(-1).float() / 10.0
        x = torch.cat([flat_state, strategy_emb, game_type_scalar], dim=-1)
        return self.net(x)

    def load_pretrained(self, path: str):
        self.load_state_dict(torch.load(path, map_location='cpu'))
        self.eval()

    def save(self, path: str):
        torch.save(self.state_dict(), path)


def create_default_policy() -> MCPolicy:
    return MCPolicy()
