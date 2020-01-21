'''A representation of the 'cyclops' game.

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

NUM_COINS = 100
NUM_ROUNDS = 7


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

  def is_terminal(self):
    '''Returns true if the game is over.'''
    return (np.abs(self.delta_num_rounds_won) > self.num_rounds_left or
            self.num_rounds_left == 0)

  def _get_player_actions(self, num_coins_left):
    if self.num_rounds_left == 1:
      return [num_coins_left]
    return range(num_coins_left + 1)

  def get_actions_player1(self):
    '''Returns the list of actions available to player 1.

    The order of the actions is guaranteed to be the same across multiple calls.
    '''
    return self._get_player_actions(self.num_coins_left_player1)

  def get_actions_player2(self):
    '''Returns the list of actions available to player 2.

    The order of the actions is guaranteed to be the same across multiple calls.
    '''
    return self._get_player_actions(self.num_coins_left_player2)

  def score(self):
    '''Returns the score of a terminal state for player 1.

    Wins are 1, losses are -1.
    '''
    if not self.is_terminal():
      return ValueError('State {} is not terminal'.format(self))
    return np.sign(self.delta_num_rounds_won)
