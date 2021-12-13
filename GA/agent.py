import os
import sys
sys.path.append(os.pardir)

import numpy as np
import random
from iagent import IAgent
from geister import Geister
from geister2 import Geister2


class GAgent(IAgent):
    LEN_CHROM = 64

    def __init__(self, game, seed=42, chrom=None):
        super(GAgent, self).__init__(game, seed)
        np.random.seed(seed)

        if chrom is None:
            self.chrom = np.random.rand(GAgent.LEN_CHROM)
        else:
            self.chrom = chrom


    def init_red(self):
        return ['A', 'B', 'C', 'D']
        root = game
        init_place = None
        scores = np.array(self.evaluate(ip) for ip in init_place)
        return init_place[scores.argmax()]



    def evaluate(self):
        return random.randint(10)



if __name__ == '__main__':
    game = Geister2()

    agent1 = GAgent(game)
    agent2 = GAgent(game)
    
    game.printBoard()

    game.setRed(agent1.init_red())
    game.changeSide()
    game.setRed(agent2.init_red())
    game.changeSide()

    game.printBoard()

