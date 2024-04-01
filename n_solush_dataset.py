import random

import torch
from torch.utils.data import Dataset
from rubarrel import Rubarrel, Player

class RubarrelDataset(Dataset):
    def __init__(self, n_samples, n_actions):
        self.n_samples = n_samples
        self.n_actions = n_actions
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
            valid_choices = set(range(0, self.n_actions))

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
        player = Player(None, moves)
        state = player.play()
        solush = torch.tensor(list(reversed(list(moves))))

        return state, solush
data = RubarrelDataset()
print(data[3])