import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.minecraft.mc_policy import MCPolicy
import numpy as np

policy = MCPolicy()
state = np.random.randn(141).astype(np.float32)
action = policy.get_action(state, 'GATHER_WOOD')
print(f'Action shape: {action.shape}, range: [{action.min():.2f}, {action.max():.2f}]')

os.makedirs('checkpoints', exist_ok=True)
policy.save('checkpoints/pretrained_policy.pt')
print('Pre-trained policy saved to checkpoints/pretrained_policy.pt')
