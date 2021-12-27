import os, sys
sys.path.append(os.pardir)

import numpy as np
import random
import itertools
from tqdm import tqdm
from iagent import IAgent
from geister2 import Geister2


class GAgent(IAgent):
    LEN_CHROM = sum(range(42*3 + 1)) # 8001

    def __init__(self, game, seed=42, chrom=None):
        super(GAgent, self).__init__(game, seed)
        np.random.seed(seed)

        if chrom is None:
            self.chrom = np.random.rand(GAgent.LEN_CHROM) * 2 - 1
        else:
            assert len(chrom) == GAgent.LEN_CHROM
            self.chrom = chrom

        self.ROW_IDS = []
        self.COL_IDS = []
        for i in range(42*3):
            self.ROW_IDS += [i]*(i+1)
            for j in range(i+1):
                self.COL_IDS += [j]


    def init_red(self):
        units = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        red_places = list(itertools.combinations(units, 4))

        states = []
        g = self._game
        for red_place in red_places:
            g.setRed(red_place)
            states.append(self.relate_pairwise(g.crr_state()))

        scores = np.array([self.evaluate(s) for s in states])
        return red_places[scores.argmax()]

    
    def get_act_afterstates(self, states):
        rps = [self.relate_pairwise(s) for s in states]
        scores = [self.evaluate(rp) for rp in rps]
        nxt = np.array(scores).argmax()
        return nxt


    def relate_pairwise(self, state):
        s = np.array(state)
        x1, x2 = s.reshape(1, -1), s.reshape(-1, 1)
        m = np.triu(x1 * x2)
        return m.reshape(-1)[:GAgent.LEN_CHROM]


    def evaluate(self, x):
        return np.dot(x, self.chrom)



if __name__ == '__main__':
    game = Geister2()

    chroms = np.load('weights/latest_100.npy', allow_pickle=True)
    agents = [GAgent(game, chrom=chroms[0][0]), GAgent(game, chrom=chroms[1])]
    
    for agent in agents:
        game.setRed(agent.init_red())
        game.changeSide()

    #game.printBoard()

    res = [0]*3
    for i in tqdm(range(100)):
        player = 0
        turn = 0
        while game.checkResult() == 0 and turn < 150:
            states = game.after_states()
            act = agents[player].get_act_afterstates(states)
            game.on_action_number_received(act)
            game.changeSide()
            player ^= 1
            turn += 1
        res[player^1 if game.checkResult() != 0 else 2] += 1
        #game.changeSide()
        #game.printAll()

    print(res)


