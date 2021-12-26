import os, sys
sys.path.append(os.pardir)

import argparse
import bisect
import numpy as np
import concurrent.futures
from tqdm import tqdm
from numpy.random import default_rng
rng = default_rng()

from geister2 import Geister2
from agent import GAgent
from random_agent import RandomAgent
from greedy_agent import GreedyAgent

pop_size = 32
len_chrom = GAgent.LEN_CHROM


def main(chroms, n_gen, n_gengap = 25, mut_pb=0.01):
    for gen in tqdm(range(n_gen)):
        scores = round_robin(chroms)
        print(scores)

        rank_idx = scores.argsort()[::-1] 
        chroms = chroms[rank_idx]
        scores = scores[rank_idx]

        if gen == n_gen - 1:
            return chroms

        if gen % 10 == 0:
            test(chroms[0], 'q-learn', 100)

        n_parents = pop_size - n_gengap
        n_crossov = pop_size - n_parents
        n_elite = 2
        n_roulette = n_parents - n_elite

        # elite select
        parents = np.zeros([n_parents, len_chrom])
        parents[:n_elite] = chroms[:n_elite]

        chroms = chroms[n_elite:]
        scores = scores[n_elite:]

        # roulette select
        p = scores - scores.min() + 1
        select_p = p / p.sum()
        parents[n_elite:] = rng.choice(chroms, n_roulette, replace=False, p=select_p)

        nxt_chroms = np.zeros([pop_size, len_chrom])

        for i in range(n_crossov):
            c1, c2 = rng.choice(parents, 2, replace=False)
            nxt_chroms[i] = cross_over(c1, c2)

        # mutation
        for i in range(n_crossov):
            if np.random.rand() < mut_pb:
                mutate(nxt_chroms[i])

        nxt_chroms[n_crossov:] = parents

        chroms = nxt_chroms


def round_robin(chroms):
    with concurrent.futures.ProcessPoolExecutor() as e:
        results = [r for r in e.map(_round_robin, range(pop_size), [chroms]*pop_size)]
    return np.array(results).sum(axis=0)

def _round_robin(idx, chroms):
    results = np.zeros(pop_size, dtype=np.int32)
    for j in range(pop_size):
        i = idx
        if i == j: continue

        winner = battle(chroms[i], chroms[j])
        if winner is not None:
            if winner == 1:
                i, j = j, i
            results[i] += 1
            results[j] -= 1
    return results


def battle(c1, c2, p2='chrom'):
    game = Geister2()
    agent1 = GAgent(game, chrom=c1)
    if p2 == 'chrom':
        agent2 = GAgent(game, chrom=c2)
    elif p2 == 'random':
        agent2 = RandomAgent(game, seed=np.random.randint(10**3))
    elif p2 == 'q-learn':
        agent2 = GreedyAgent(game)
        agent2.theta = np.load('../weights/vsself2/vsself900_theta.npy')

    agents = [agent1, agent2]

    for agent in agents:
        game.setRed(agent.init_red())
        game.changeSide()

    p = 0
    turn = 0
    while game.checkResult() == 0 and turn < 150:
        states = game.after_states()
        act = agents[p].get_act_afterstates(states)
        game.on_action_number_received(act)
        game.changeSide()
        p ^= 1
        turn += 1

    return p ^ 1 if game.checkResult() != 0 else None


def cross_over(c1, c2):
    c = np.zeros(GAgent.LEN_CHROM)
    for i in range(GAgent.LEN_CHROM):
        c[i] = c1[i] if np.random.rand() < 0.5 else c2[i]
    return c


def mutate(c):
    n = len_chrom // 100
    idx = np.random.randint(0, len_chrom, n)
    c[idx] = np.random.rand(n)*2 - 1


def test(c, agent, n):
    results = [0]*3
    for i in tqdm(range(n)):
        r = battle(c, None, p2=agent)
        if r is None:
            results[2] += 1
        else:
            results[r] += 1
    print('vs {}, Win:{} Lose:{} Draw:{}'.format(agent, *results))
    return results




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--chroms', type=str, default='random')
    parser.add_argument('-g', '--n-gen', type=int, default=100)
    parser.add_argument('-o', '--output', type=str, default='latest')
    args = parser.parse_args()


    if args.chroms == 'random':
        chroms = np.random.rand(pop_size, GAgent.LEN_CHROM)*2 - 1
    else:
        chroms = np.load(args.chroms)

    
    test(chroms[0], 'q-learn', 100)

    chroms = main(chroms, args.n_gen)

    test(chroms[0], 'q-learn', 100)

    np.save('weights/' + args.output + '_' + str(args.n_gen), chroms)

