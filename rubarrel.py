import numpy as np
import torch


class Rubarrel():
    def __init__(self, state_vec=None):
        if state_vec is None:
            self.rows = []
            for i in range(5):
                self.rows.append([])
                for _ in range(4):
                    self.rows[i].append(i)
            self.out = [5, 5, 5]
            self.side = "left"
        else:
            self.rows = []
            for i in range(5):
                self.rows.append(list(state_vec[i * 4:i * 4 + 4]))
            self.out = state_vec[20:23]
            self.side = "right" if state_vec[-1] else "left"

    def __repr__(self):
        repr = ""
        for i, row in enumerate(self.rows):
            out = [self.out[0], " ", self.out[1], self.out[2], " "]
            r = 5 * [" "]
            l = 5 * [" "]
            if self.side == "left":
                l = out
            else:
                r = out
            repr += str(l[i]) + str(row) + str(r[i]) + "\n"
        return repr

    def turn(self, side, amount):
        if side == "left":
            idx = [0, 1]
        if side == "right":
            idx = [2, 3]

        new_rows = []
        for i, row in enumerate(self.rows):
            new_rows.append([])
            for j, entry in enumerate(row):
                elem = self.rows[(i + amount) % 5][j] if j in idx else self.rows[i][j]
                new_rows[i].append(elem)
        self.rows = new_rows

    def shift(self):
        m = {0: 0, 2: 1, 3: 2}
        for i, row in enumerate(self.rows):
            if i in [0, 2, 3]:
                if self.side == "left":
                    self.rows[i] = [self.out[m[i]]] + row[:-1]
                    self.out[m[i]] = row[-1]
                elif self.side == "right":
                    self.rows[i] = row[1:] + [self.out[m[i]]]
                    self.out[m[i]] = row[0]
        self.side = "left" if self.side == "right" else "right"

    def solved(self):
        for row in self.rows:
            if not len(set(row)) == 1:
                return False
        if not self.out == [5, 5, 5]:
            return False
        return True

    def get_state(self):
        state = []
        for row in self.rows:
            state += row
        state += self.out
        if self.side == "left":
            state.append(0)
        elif self.side == "right":
            state.append(1)
        return torch.tensor(state)


class Player:
    def __init__(self, state_vec, moves = None):
        self.barrel = Rubarrel(state_vec)
        self.state_vec = state_vec
        if moves is not None:
            self.moves = moves
        else:
            self.moves = [-1]

        self.move_dict = {-1: "nothing",
                          0: "shift",
                          1: ("left", 1),
                          2: ("right", 1),
                          3: ("left", -1),
                          4: ("right", -1),
                          5: ("left", 2),
                          6: ("right", 2),
                          7: ("left", -2),
                          8: ("right", -2)}

    def play(self, moves = None):
        self.barrel = Rubarrel(self.state_vec)
        if moves is not None:
            self.moves = moves
        for move in self.moves:
            print("d", self.barrel.out)
            try:
                self.make_move(move.item())
            except:
                self.make_move(move)
        self.state_vec = self.barrel.get_state()
        return self.state_vec

    def make_move(self, m_id):
        if self.move_dict[m_id] == "shift":
            self.barrel.shift()
        elif self.move_dict[m_id] == "nothing":
            pass
        else:
            self.barrel.turn(*(self.move_dict[m_id]))


state_vec = torch.tensor([0, 0, 0, 0,
                      1, 1, 1, 1,
                      2, 2, 2, 2,
                      3, 3, 3, 3,
                      4, 4, 4, 4,
                      5, 5, 5,
                      0])
moves = torch.tensor([0,3])
player = Player(state_vec)
print(player.play())
print(player.barrel.get_state())
print(player.play(moves))
print(player.play([1,0]))
