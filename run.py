import client
import argparse
import tcp_player
import numpy as np
from geister2 import Geister2
from GA.agent import GAgent

def main(args):
    game = Geister2()

    agents = {
        'GA': GAgent
    }
    chrom = np.load('GA/weights/latest_100.npy', allow_pickle=True)
    agent = agents[args.agent](game, chrom=chrom[1])

    res = tcp_player.tcp_connect(agent, game, args.port, args.host, args.n_games)
    print('WIN: {}\nDRAW: {}\nLOSE: {}'.format(*res))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('host', type=str)
    parser.add_argument('-p', '--port', type=int, default=10000)
    parser.add_argument('-a', '--agent', type=str, default='GA')
    parser.add_argument('-n', '--n-games', type=int, default=1)

    main(parser.parse_args())

