import os, sys
sys.path.append(os.pardir)

import numpy as np
from tqdm import tqdm
from numpy.random import default_rng
rng = default_rng()

from geister2 import Geister2


def battle(agent1, agent2, a1=None, a2=None):
    game = Geister2()
    agents = [agent1(game, a1), agent2(game, a2)]

    for agent in agents:
        game.setRed(agent.init_red())
        game.changeSide()

    p = 0
    for _ in range(150):
        states = game.after_states()
        i_act = agents[p].get_act_afterstates(states)
        game.on_action_number_received(i_act)
        game.changeSide()
        p ^= 1
        if game.is_ended():
            if p == 1: game.changeSide()
            return int(game.checkResult() < 0)
    return -1

