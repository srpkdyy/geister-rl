import os, sys
sys.path.append(os.pardir)

import numpy as np
rng = np.random.default_rng()
import random
import itertools
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from iagent import IAgent
from geister2 import Geister2


class GAgent(IAgent):
    LEN_CHROM = sum(range(42*3 + 1)) # 8001

    def __init__(self, game, chrom=None, seed=None):
        super(GAgent, self).__init__(game, seed)
        np.random.seed(seed)

        if chrom is None:
            self.chrom = rng.uniform(-10, 10, GAgent.LEN_CHROM)
        else:
            assert len(chrom) == GAgent.LEN_CHROM
            self.chrom = chrom

        self.PAIR_IDX = np.tril_indices(42*3)


    def init_red(self):
        units = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        red_places = list(itertools.combinations(units, 4))
        return red_places[np.random.randint(len(red_places))]
        '''
        g = Geister2()
        pw = self.relate_pairwise
        rps = [pw(g.setRed(p).crr_state()) for p in red_places]
        scores = [self.evaluate(rp) for rp in rps]
        i_place = np.array(scores).argmax()
        return red_places[i_place]
        '''

    
    def get_act_afterstates(self, states):
        rps = [self.relate_pairwise(s) for s in states]
        scores = [self.evaluate(rp) for rp in rps]
        scores = np.array(scores)
        p = np.exp(scores)
        p /= p.sum()
        #scores = np.array(scores) - min(scores)
        #scores /= scores.sum()
        nxt = np.random.choice(len(states), 1, p=p)[0]

        #nxt = scores.argmax()
        return nxt


    def relate_pairwise(self, state):
        s = np.array(state)
        x = s.reshape(-1)
        m = x[self.PAIR_IDX[0]] * x[self.PAIR_IDX[1]]
        return m


    def evaluate(self, x):
        return np.dot(x, self.chrom)



if __name__ == '__main__':
    import utils
    chroms = np.load('weights/best.npy', allow_pickle=True)

    result = [0]*3
    n = 100
    for i in tqdm(range(n)):
        r = utils.battle(GAgent, RandomAgent, chroms, None)
        result[r] += 1

    print('Win:{}, Lose:{}, Draw:{}'.format(*result))

