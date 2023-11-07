
from cap6635.environment.map import Map2D
from cap6635.utilities.plot import MazeAnimator
from cap6635.utilities.constants import (
    MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT
)
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from agents import MazeRunner

get_action = {MOVE_UP: '^', MOVE_DOWN: 'v', MOVE_LEFT: '<', MOVE_RIGHT: '>'}
hw1_usage = (
    'Usage:\n'
    'python3 hw1_dynamic_programming.py <elements_in_row> <algorithm>\n'
    '     <deterministic(y/n)> <simulation(y/n)> [starting_state]\n\n'
    'algorithms to choose from:\n'
    '(v) Value Iteration\n'
    '(p) Policy Iteration\n'
    '(e) Policy Evaluation\n')
hw2_usage = (
    'Usage:\n'
    'python3 hw2_sarsa.py <elements_in_row> <algorithm>\n'
    '     <deterministic(y/n)> <simulation(y/n)> [starting_state]\n\n'
    'algorithms to choose from:\n'
    '(q) Q-Learning\n'
    '(s) SARSA\n'
    'NOTE: deterministic is the only supported option.')


def pretty_policy(P, elements_in_row, E, B, text):
    policy = Map2D(elements_in_row + 2, elements_in_row + 2)
    for i, s in enumerate(P):
        action = np.where(s == 1)[0][0]
        y = (i % elements_in_row) + 1
        x = (math.floor(i / elements_in_row)) + 1
        if i in E:
            plt.text(y, x, "x", color='green')
        elif i in B:
            plt.text(y, x, "o", color='green')
        else:
            plt.text(y, x, get_action[action], color='green')
        plt.imshow(policy.map, 'pink')
    plt.savefig('policy_%s.png' % (text))
    plt.clf()


def gather_inputs(usage):
    # States: size of your Grid (|column| * |row|)
    try:
        elements_in_row = int(sys.argv[1])
    except ValueError:
        elements_in_row = 10
    except IndexError:
        elements_in_row = 10

    try:
        algo = sys.argv[2]
    except IndexError:
        algo = 'v'
    try:
        if sys.argv[3] == 'n':
            deterministic = False
        else:
            deterministic = True
    except BaseException:
        print(usage)
        sys.exit(1)

    try:
        if sys.argv[4] == 'n':
            gui = False
        else:
            gui = True
            try:
                start_pos = int(sys.argv[5])
            except ValueError:
                start_pos = 0
    except BaseException:
        print(usage)
        sys.exit(1)

    return elements_in_row, algo, deterministic, gui, start_pos


def animate(start_pos, E, B, elements_in_row, algo, P, maze_name):
    maze = Map2D(elements_in_row + 2, elements_in_row + 2)
    for e in E:
        maze.map[
            (math.floor(e / elements_in_row)) + 1][
            (e % elements_in_row) + 1
            ] = 20
    for b in B:
        maze.map[
            (math.floor(b / elements_in_row)) + 1][
            (b % elements_in_row) + 1
            ] = 5

    sx = math.floor(start_pos / elements_in_row) + 1
    sy = math.floor(start_pos % elements_in_row) + 1
    a = MazeRunner(maze, P, state=start_pos, start=(sx, sy),
                   elements_in_row=elements_in_row)
    i = 0
    animator = MazeAnimator(os.getcwd(), maze_name)
    animator.temp = '/temp/'
    animator.save_state(i, maze, a)
    while a.state not in E:
        # Starting position
        a.move()
        maze.map[a._x_path[0], a._y_path[0]] = 30
        maze.map[a._x_path[-1], a._y_path[-1]] = 10
        maze.map[a._x_path[-2], a._y_path[-2]] = 0
        animator.save_state(i, maze, a)
        i += 1

    animator.make_gif()
    del animator.temp
