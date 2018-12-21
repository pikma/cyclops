'''An AI to play the 'cyclops' game.

This is a 2 player game. Each player has n coins, and there are m rounds. In
each round, both player simulatenously decide to spend some of their coins
(they can spend 0). The player who spends the most coins wins the round. If
they spend the same amount, the round is a tie. After m rounds, the player who
has won the most rounds wins the game.

This file implements the backwards-induction algorithm, see 'Algorithms for
Computing Strategies in Two-Player Simultaneous Move Games', by Bosansky et
al (http://mlanctot.info/files/papers/aij-2psimmove.pdf).
'''

import collections

import numpy as np
from ortools.linear_solver import pywraplp

NUM_COINS = 10
NUM_ROUNDS = 3


class State(
    collections.namedtuple('State', [
        'num_coins_left_player1', 'num_coins_left_player2',
        'delta_num_rounds_won', 'num_rounds_left'
    ])):
  '''A state in the game.

  The state captures all the information necessary to play: the number of coins
  remaining and the numbers of rounds won for both players, as well as the
  number of ties.
  '''
  __slots__ = ()

  @staticmethod
  def initial_state():
    '''Creates the initial state of the game.'''
    return State(
        num_coins_left_player1=NUM_COINS,
        num_coins_left_player2=NUM_COINS,
        delta_num_rounds_won=0,
        num_rounds_left=NUM_ROUNDS)

  def __str__(self):
    return '{: <2} | {: <2} | {:+d}, {}'.format(
        self.num_coins_left_player1, self.num_coins_left_player2,
        self.delta_num_rounds_won, self.num_rounds_left)

  def next_state(self, num_coins_spent_player1, num_coins_spent_player2):
    '''Returns the state of the game after both players play a round.'''
    if self.num_coins_left_player1 < num_coins_spent_player1:
      raise ValueError('Player 1 cannot spend {} coins in state {}'.format(
          num_coins_spent_player1, self))
    if self.num_coins_left_player2 < num_coins_spent_player2:
      raise ValueError('Player 2 cannot spend {} coins in state {}'.format(
          num_coins_spent_player2, self))

    return State(
        num_coins_left_player1=self.num_coins_left_player1 -
        num_coins_spent_player1,
        num_coins_left_player2=self.num_coins_left_player2 -
        num_coins_spent_player2,
        delta_num_rounds_won=self.delta_num_rounds_won + int(
            np.sign(num_coins_spent_player1 - num_coins_spent_player2)),
        num_rounds_left=self.num_rounds_left - 1)

  def is_terminal_state(self):
    '''Returns true if the game is over.'''
    return self.num_rounds_left == 0

  def _get_player_actions(self, num_coins_left):
    if self.num_rounds_left == 1:
      return [num_coins_left]
    return range(num_coins_left + 1)

  def get_actions_player1(self):
    '''Returns the list of actions available to player 1.

    The order of the actions is guaranteed to be the same across multiple calls.
    '''
    return self._get_player_actions(self.num_coins_left_player1)

  def get_actoins_player2(self):
    '''Returns the list of actions available to player 2.

    The order of the actions is guaranteed to be the same across multiple calls.
    '''
    return self._get_player_actions(self.num_coins_left_player2)

  def score(self):
    '''Returns the score of a terminal state for player 1.

    Wins are 1, losses are -1.
    '''
    if not self.is_terminal_state():
      return ValueError('State {} is not terminal'.format(self))
    return np.sign(self.delta_num_rounds_won)


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


class StrategyBook(object):
  '''A book of best strategy for each game state.'''

  def __init__(self):
    self._strategies = {}

  def get_value(self, state):
    item = self._strategies.get(state)
    if item:
      return item[1]
    return None

  def set_strategy(self, state, strategy, value):
    self._strategies[state] = (strategy, value)

  def __str__(self):
    strategies_str = []

    def _sorting_order(state):
      return (-state.num_rounds_left, np.abs(state.delta_num_rounds_won) +
              state.delta_num_rounds_won / (NUM_ROUNDS + 1),
              -state.num_coins_left_player1, -state.num_coins_left_player2)

    sorted_strategies = sorted(
        self._strategies.items(), key=lambda x: _sorting_order(x[0]))

    for state, (strategy, value) in sorted_strategies:
      if len(strategy) <= 1:
        continue
      strategy_str = str(state) + ': ['
      actions = state.get_actions_player1()
      for action_ix, prob in enumerate(strategy):
        if action_ix != 0:
          strategy_str += ', '
        strategy_str += '{}: {:.2g}'.format(actions[action_ix], prob)
      strategy_str += '], value = {:.2g}'.format(np.round(value, 3))
      strategies_str.append(strategy_str)
    return '\n'.join(strategies_str)


def backwards_induction(state, strategy_book):
  '''
  Computes the optimal strategy for player1 in this state.

  Returns the value of the state for player 1.
  '''
  if state.is_terminal_state():
    return state.score()

  value = strategy_book.get_value(state)
  if value:
    return value

  player1_actions = state.get_actions_player1()
  player2_actions = state.get_actoins_player2()

  matrix = np.zeros([len(player1_actions), len(player2_actions)])

  for i, player1_action in enumerate(player1_actions):
    for j, player2_action in enumerate(player2_actions):
      next_state = state.next_state(player1_action, player2_action)
      matrix[i, j] = backwards_induction(next_state, strategy_book)

  try:
    strategy, value = solve_matrix_game(matrix)
  except Exception as e:
    raise Exception('Optimization failed for state {}: {}'.format(state, e))
  strategy_book.set_strategy(state, strategy, value)
  return value


def main():
  strategy_book = StrategyBook()
  initial_state = State.initial_state()

  backwards_induction(initial_state, strategy_book)

  print(strategy_book)


if __name__ == '__main__':
  main()
