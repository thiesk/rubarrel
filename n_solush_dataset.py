import random

import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from player import Player

class RubarrelDataset(Dataset):
    def __init__(self, n_samples, n_actions, idx_action=4):
        self.n_samples = n_samples
        self.n_actions = n_actions
        self.idx_action = idx_action
        self.constraints = {
            0: [0],
            1: [3],
            3: [1],
            2: [4],
            4: [2],
        }

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        torch.manual_seed(idx)
        sequence = []
        last_choice = None

        for _ in range(self.n_actions):
            valid_choices = set(range(0, self.idx_action))

            # Remove choices that violate constraints
            if last_choice is not None and last_choice in self.constraints:
                valid_choices -= set(self.constraints[last_choice])

            # Ensure there are valid choices left
            if not valid_choices:
                raise ValueError("No valid choices available due to constraints.")

            # Choose a random element from the valid choices
            choice = random.choice(list(valid_choices))
            sequence.append(choice)
            last_choice = choice

        moves = torch.tensor(sequence)
        player = Player()
        player.play(moves)
        state = player.barrel.get_state()
        solush = moves
        for i, m_id in enumerate(reversed(moves)):
            m_id = int(m_id)
            if m_id == 0:
                solush[i] = 0
            elif m_id == 1:
                solush[i] = 3
            elif m_id == 2:
                solush[i] = 4
            elif m_id == 3:
                solush[i] = 1
            elif m_id == 4:
                solush[i] = 2


        return state, solush
data = RubarrelDataset(10000,10)

for i in range(1000):
    state, solush = data[i]
    player = Player(state)
    player.play(solush)
    if not player.barrel.solved():
        print(solush)
print(data[2][0].shape, data[2][1].shape)