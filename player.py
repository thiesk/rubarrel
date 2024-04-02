import torch
from rubarrel import Rubarrel
class Player:
    def __init__(self, state_vec=None):
        self.barrel = Rubarrel(state_vec)
        self.move_dict = {-1: "nothing",
                          0: "shift",
                          1: {"side": "left", "amount": 1},
                          2: {"side": "right", "amount": 1},
                          3: {"side": "left", "amount": -1},
                          4: {"side": "right", "amount": -1},
                          5: {"side": "left", "amount": 2},
                          6: {"side": "right", "amount": 2},
                          7: {"side": "left", "amount": -2},
                          8: {"side": "right", "amount": -2}}

    def play(self, moves):
        for move in moves:
            move = int(move)
            if move == -1:
                pass
            elif move == 0:
                self.barrel.shift()
            else:
                self.barrel.turn(**(self.move_dict[move]))

