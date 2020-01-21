'''
This file implements the backwards-induction algorithm, see 'Algorithms for
Computing Strategies in Two-Player Simultaneous Move Games', by Bosansky et
al (http://mlanctot.info/files/papers/aij-2psimmove.pdf).
'''

import argparse

import numpy as np
from ortools.linear_solver import pywraplp

from strategy_book import StrategyBook
import cyclops
import strategy_pb2


def solve_matrix_game(matrix):
  '''Finds the optimal strategy of a matrix game, for player 1.

  Returns (strategy, value), where strategy is the list of probabilities of each
  action available to player 1.
  '''
  # Curiously, the optimizer struggles with very small numbers: it fails to find
  # solutions to the constraints, even though they are trivial. We round the
  # matrix to solve this issue, since the precision does not matter much.
  matrix = np.round(matrix, 12)

  solver = pywraplp.Solver('SolveMatrixGame',
                           pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

  # Set up the variables.
  value = solver.NumVar(-1, 1, 'value')
  action_probs = []
  for i in range(matrix.shape[0]):
    action_probs.append(solver.NumVar(0, 1, 'p_{}'.format(i)))

  # Define the objectives.
  objective = solver.Objective()
  objective.SetCoefficient(value, 1)
  objective.SetMaximization()

  # Set the constraints.
  #
  # First constraint: The probabilities must sum to 1.
  sum_prob_constraint = solver.Constraint(1, 1)
  for action_prob in action_probs:
    sum_prob_constraint.SetCoefficient(action_prob, 1)

  # Second constraint: The value of the game is defined as:
  #   v = max_pi min_pj (pi . A . pj)
  #     = max_pi min_j (pi . A_j)
  #
  # This is equivalent to maximizing v such that forall j, pi . A_j >= v.
  for player2_action in range(matrix.shape[1]):
    constraint = solver.Constraint(0, solver.infinity())
    for player1_action in range(matrix.shape[0]):
      action_prob = action_probs[player1_action]
      constraint.SetCoefficient(action_prob,
                                matrix[player1_action, player2_action])
    constraint.SetCoefficient(value, -1)

  solver_result = solver.Solve()
  if solver_result not in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
    raise Exception('Solver returned {}, matrix = {}'.format(
        solver_result, matrix))

  return ([p.solution_value() for p in action_probs], value.solution_value())


def backwards_induction(initial_state, strategy_book, verbose=False):
  '''
  Computes the optimal strategy for player1.

  Computes both the optimal strategy and the value of all states starting from
  the initial_state, and populates the strategies and values of each state in
  the strategy_book.
  '''

  num_states_to_process = None
  num_states_processed = 0
  if verbose:
    num_states_to_process = num_non_terminal_states(initial_state)
    print('Game has {} states'.format(num_states_to_process))

  def _get_value(state, strategy_book):
    if state.is_terminal():
      return state.score()
    return strategy_book.get_value(state)

  visited_states = [initial_state]

  while visited_states:
    state = visited_states[-1]

    if _get_value(state, strategy_book) is not None:
      visited_states.pop()
      continue

    player1_actions = state.get_actions_player1()
    player2_actions = state.get_actions_player2()

    matrix = np.zeros([len(player1_actions), len(player2_actions)])

    all_next_states_processed = True

    for i, player1_action in enumerate(player1_actions):
      for j, player2_action in enumerate(player2_actions):
        next_state = state.next_state(player1_action, player2_action)
        value = _get_value(next_state, strategy_book)
        if value is not None:
          matrix[i, j] = value
        else:
          visited_states.append(next_state)
          all_next_states_processed = False

    if all_next_states_processed:
      visited_states.pop()
      num_states_processed += 1
      try:
        strategy, value = solve_matrix_game(matrix)
      except Exception as e:
        raise Exception('Optimization failed for state {}: {}'.format(state, e))
      strategy_book.set_strategy(state, strategy, value)

    if verbose:
      print('Processed {} / {} states'.format(num_states_processed,
                                              num_states_to_process))


def num_non_terminal_states(initial_state):
  '''
  Computes the total number of states following an initial_state.
  '''
  visited = [initial_state]
  processed = set()

  while visited:
    current_state = visited.pop()
    if current_state.is_terminal() or current_state in processed:
      continue

    processed.add(current_state)

    for player1_action in current_state.get_actions_player1():
      for player2_action in current_state.get_actions_player2():
        next_state = current_state.next_state(player1_action, player2_action)
        visited.append(next_state)

  return len(processed)


def main():
  parser = argparse.ArgumentParser(description='Computes strategies.')
  parser.add_argument(
      '--strategy_file_out',
      type=str,
      help='path to the output file for the strategy book')

  args = parser.parse_args()

  strategy_book = StrategyBook()
  initial_state = cyclops.State.initial_state()

  backwards_induction(initial_state, strategy_book, verbose=True)
  serialized_strategy = strategy_book.serialize()

  print(
      'Initial strategy: {}'.format(StrategyBook.strategy_to_str(
        initial_state, strategy_book.get_strategy(initial_state))))

  game_value = strategy_book.get_value(initial_state)
  if np.abs(game_value) > 0.0001:
    raise Exception(
        'Value found for initial state: {}, should be 0'.format(game_value))
  output_file = args.strategy_file_out
  with open(output_file, 'wb') as f:
    f.write(serialized_strategy)


if __name__ == '__main__':
  main()
