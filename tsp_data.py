import math
import numpy as np
import random
import itertools


class Tsp:
    def next_batch(self, batch_size=1):
        X, Y = [], []
        for b in range(batch_size):
            print("preparing dataset... %s/%s" % (b, batch_size))
            points = self.generate_data()
            solved = self.solve_tsp_dynamic(points)
            X.append(points), Y.append(solved)
        return np.asarray(X), np.asarray(Y)

    def length(self, x, y):
        return (math.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2))

    def solve_tsp_dynamic(self, points):
        # calc all lengths
        all_distances = [[self.length(x, y) for y in points] for x in points]
        # initial value - just distance from 0 to
        # every other point + keep the track of edges
        A = {(frozenset([0, idx+1]), idx+1): (dist, [0, idx+1])
             for idx, dist in enumerate(all_distances[0][1:])}
        cnt = len(points)
        for m in range(2, cnt):
            B = {}
            for S in [frozenset(C) | {0}
                      for C in itertools.combinations(range(1, cnt), m)]:
                for j in S - {0}:
                    B[(S, j)] = min([(A[(S-{j}, k)][0] + all_distances[k][j],
                                      A[(S-{j}, k)][1] + [j])
                                     for k in S if k != 0 and k != j])
            A = B
        res = min([(A[d][0] + all_distances[0][d[1]], A[d][1])
                   for d in iter(A)])
        return res[1]

    def generate_data(self, N=10):
        radius = 1
        rangeX = (0, 10)
        rangeY = (0, 10)
        qty = N

        deltas = set()
        for x in range(-radius, radius+1):
            for y in range(-radius, radius+1):
                if x*x + y*y <= radius*radius:
                    deltas.add((x, y))

        randPoints = []
        excluded = set()
        i = 0
        while i < qty:
            x = random.randrange(*rangeX)
            y = random.randrange(*rangeY)
            if (x, y) in excluded:
                continue
            randPoints.append((x, y))
            i += 1
            excluded.update((x+dx, y+dy) for (dx, dy) in deltas)
        return randPoints

if __name__ == "__main__":
    p = Tsp()
    X, Y = p.next_batch(1)
    print(X)
    print(Y)
