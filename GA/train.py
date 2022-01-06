import os, sys
sys.path.append(os.pardir)

import argparse
import bisect
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from numpy.random import default_rng
rng = default_rng()

from geister2 import Geister2
from agent import GAgent
from random_agent import RandomAgent
from greedy_agent import GreedyAgent
from load_ import load_agent

pop_size = 1297
len_chrom = GAgent.LEN_CHROM
init_max = 10
init_min = -10


def main(chroms, n_gen, n_gengap = 500, mut_pb=0.01):
    n_parents = pop_size - n_gengap
    n_children = pop_size - n_parents
    n_elite = 48
    n_roulette = n_parents - n_elite
    #n_ranking = n_parents - n_elite

    most_elite = None
    max_win = 0

    for gen in tqdm(range(n_gen)):
        scores = get_fitness(chroms)
        print(scores)
        print('Mean: {} Max: {} Min: {}'.format(scores.mean(), scores.max(), scores.min()))

        rank_idx = scores.argsort()[::-1] 
        chroms = chroms[rank_idx]
        scores = scores[rank_idx]

        if gen == n_gen - 1:
            return chroms, most_elite

        if gen % 10 == 0:
            print(chroms)
            n_tests = 100
            win, *_ = test(chroms[0], 'q-learn', n_tests)
            if win > max_win:
                most_elite = chroms[0]
                max_win = win
            if max_win == n_tests:
                return chroms, most_elite
            


        # elite select
        parents = np.zeros([n_parents, len_chrom])
        parents[:n_elite] = chroms[:n_elite]
        #parents[n_elite//2:n_elite] = chroms[:n_elite//2]
        #for i in range(n_parents):
        #    mutate(parents[i+n_parents//2])

        chroms = chroms[n_elite:]
        scores = scores[n_elite:]

        # roulette select
        p = scores - scores.min() + 0.01
        select_p = p / p.sum()
        #r = np.array(range(len(chroms))[::-1]) + 0.01
        #select_p = r / r.sum()
        parents[n_elite:] = rng.choice(chroms, n_roulette, replace=False, p=select_p)

        nxt_chroms = np.zeros([pop_size, len_chrom])
        nxt_chroms[:n_parents] = parents

        for i in range(n_children):
            # cross over
            p1, p2 = rng.choice(parents, 2, replace=False)
            nxt_chroms[n_parents + i] = cross_over(p1, p2)

            # mutation
            if rng.random() < mut_pb:
                mutate(nxt_chroms[n_parents + i])

        chroms = nxt_chroms


def get_fitness(chroms):
    with ProcessPoolExecutor() as e:
        results = [r for r in e.map(_vs_ai, range(pop_size), [chroms]*pop_size)]
    return np.array(results).sum(axis=0)


def _vs_ai(idx, chroms):
    results = [0]*3
    for i in range(20):
        r = battle(chroms[idx], None, 'q-learn')
        if r is None:
            results[2] += 1
        else:
            results[r] += 1
    ret = np.zeros(pop_size)
    ret[idx] = results[0] - results[1] - 0.1 * results[2]
    return ret


def _round_robin(idx, chroms):
    results = np.zeros(pop_size)
    for j in range(pop_size):
        i = idx
        if i == j: continue

        winner = battle(chroms[i], chroms[j])
        if winner is None:
            results[i] -= 0.1
            results[j] -= 0.1
        else:
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
        agent2 = RandomAgent(game, seed=None)
    elif p2 == 'q-learn':
        agent2 = load_agent('../weights/vsself2/vsself900', game=game, seed=None, AgentClass=GreedyAgent)

    agents = [agent1, agent2]

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
    return None


def cross_over(c1, c2, alpha=0.3):
    d = np.abs(np.subtract(c1, c2))
    rmin = np.fmin(c1, c2) - alpha * d
    rmax = np.fmax(c1, c2) + alpha * d
    return rng.uniform(rmin, rmax, len_chrom)


def mutate(c):
    n = np.random.randint(len_chrom / 100) + 1
    idx = rng.choice(range(len_chrom), n, replace=False)
    c[idx] = rng.uniform(init_min, init_max, n)


def test(c, agent, n):
    result = [0]*3
    with ProcessPoolExecutor() as e:
        res = [r for r in e.map(battle, [c]*n, [None]*n, [agent]*n)]
    for r in res:
        if r is None:
            result[2] += 1
        else:
            result[r] += 1
    print('vs {}, Win:{} Lose:{} Draw:{}'.format(agent, *result))
    return result


def top_save(cs, agent, n):
    score = -10**7
    model = None
    for c in tqdm(cs):
        w, l, _ = test(c, agent, n)
        s = w - l
        if s > score:
            score = s
            model = c
        print('best: {}'.format(score))
    np.save('weights/oribest', model)
    exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--chroms', type=str, default='random')
    parser.add_argument('-g', '--n-gen', type=int, default=100)
    parser.add_argument('-o', '--output', type=str, default='latest')
    args = parser.parse_args()

    if args.chroms == 'random':
        chroms = rng.uniform(init_min, init_max, [pop_size, len_chrom])
    else:
        chroms = np.load('weights/' + args.chroms + '.npy', allow_pickle=True)
        chs = rng.choice(chroms[0][1:], pop_size-2, replace=False)
        chroms = np.vstack([chroms[0][0], chs, chroms[1]])

    top_save(chroms, 'q-learn', 200)
    
    test(chroms[0], 'random', 100)
    chroms = main(chroms, args.n_gen)

    test(chroms[0][0], 'random', 100)
    test(chroms[1], 'random', 100)

    test(chroms[0][0], 'q-learn', 100)
    test(chroms[1], 'q-learn', 100)

    chroms = np.array(chroms, dtype=object)
    np.save('weights/' + args.output + '-' + str(args.n_gen), chroms)

