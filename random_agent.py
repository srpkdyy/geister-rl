import random
import numpy as np
from tqdm import tqdm
from iagent import IAgent
from geister import Geister
from geister2 import Geister2


class RandomAgent(IAgent):
    def get_act_afterstates(self, states):
        act_i = self._rnd.randrange(len(states))
        return act_i

    def get_next_action(self, moves):
        action = self._rnd.choice(moves)
        return action

    def init_red(self):
        arr = ["A", "B", "C", "D", "E", "F", "G", "H"]
        self._rnd.shuffle(arr)
        return arr[0:4]

    def __init__(self, game, seed=1):
        self._game = game
        self._rnd = random.Random(seed)


if __name__ == "__main__":
    n = 10000
    rate = 0
    turn = [0]*n
    for i in tqdm(range(n)):
        game = Geister2()
        agents = [RandomAgent(game, None), RandomAgent(game, None)]
        for agent in agents:
            game.setRed(agent.init_red())
            game.changeSide()

        p = 0
        while True:
            turn[i] += 1
            moves = game.legalMoves()
            move = agents[p].get_next_action(moves)
            game.move(*move)
            game.changeSide()
            p ^= 1
            if game.is_ended():
                if p == 1: game.changeSide()
                r = 1 if game.checkResult() > 0 else -1
                rate += r
                break
    game.printAll()
    print(rate, r)
    turn = np.array(turn)
    print(np.histogram(turn, bins=10))
    print(turn.min(), turn.mean(), turn.max())

