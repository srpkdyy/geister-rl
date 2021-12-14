import os, sys
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
            assert len(chrom) == GAgent.LEN_CHROM
            self.chrom = chrom


    def init_red(self):
        return ['A', 'B', 'C', 'D']
        init_place = None
        scores = np.array(self.evaluate(ip) for ip in init_place)
        return init_place[scores.argmax()]

    

    def get_act_afterstates(self, game):
        nxt_act, _ = self.get_hand(game)
        return nxt_act

    
    def get_hand(self, game):
        self._game = game

        moves = self._game.legalMoves()
        scores = np.zeros(len(moves))

        for i, nxt in enumerate(moves):
            game = self._game
            game.move(*nxt)
            scores[i] = self.evaluate(game.after_states())

        nxt = scores.argmax()
        return nxt, moves[nxt]


    def evaluate(self, states):
        # return board.dot(self.chrom)
        return np.random.rand()



if __name__ == '__main__':
    game = Geister2()

    agents = [GAgent(game), GAgent(game)]
    
    for agent in agents:
        game.setRed(agent.init_red())
        game.changeSide()

    game.printBoard()

    player = 0
    while game.checkResult() == 0:
        states = game.after_states()
        act = agents[player].get_act_afterstates(states)
        game.on_action_number_received(act)
        game.changeSide()
        player ^= 1
    game.changeSide()
    game.printAll()

