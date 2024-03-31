class Rubarrel():
    def __init__(self):
        self.rows = []
        for i in range(5):
            self.rows.append([])
            for _ in range(4):
                self.rows[i].append(i)
        self.out = [6, 6, 6]
        self.side = "right"

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
                if self.side == "right":
                    self.rows[i] = row[1:] + [self.out[m[i]]]
                    self.out[m[i]] = row[0]
        self.side = "left" if self.side == "right" else "right"