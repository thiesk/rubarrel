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
        old_out = self.out
        self.out = []
        old_rows = self.rows
        self.rows = []
        out_map = {0:0,2:1,3:2}
        for i, row in enumerate(old_rows):
            if i in [0,2,3]:
                if self.side == "left":
                    self.out.append(row[-1])
                    self.rows.append([old_out[out_map[i]]])
                    self.rows[i] += row[:-1]
                elif self.side == "right":
                    self.out.append(row[0])
                    self.rows.append(row[1:])
                    self.rows[i].append(old_out[out_map[i]])
            else:
                self.rows.append(row)
        self.side = "left" if self.side == "right" else "right"

    def solved(self):
        for row in self.rows:
            try:
                row = [tensor.item() for tensor in row]
            except:
                pass

            if not len(set(list(row))) == 1:
                return False
        if not (all(self.out) < 5):
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
