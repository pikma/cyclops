import numpy as np

import cyclops
import strategy_pb2


class StrategyBook(object):
  '''A book of best strategy for each game state.'''

  def __init__(self):
    self._strategies = {}

  def get_value(self, state):
    item = self._strategies.get(state)
    if item:
      return item[1]
    return None

  def get_strategy(self, state):
    item = self._strategies.get(state)
    if item:
      return item[0]
    return None

  def set_strategy(self, state, strategy, value):
    self._strategies[state] = (strategy, value)

  @staticmethod
  def strategy_to_str(state, strategy):
    result = '['
    actions = state.get_actions_player1()
    for action_ix, prob in enumerate(strategy):
      if action_ix != 0:
        result += ', '
      result += '{}: {:.2g}'.format(actions[action_ix], prob)
    result += ']'
    return result

  def __str__(self):
    strategies_str = []

    def _sorting_order(state):
      return (-state.num_rounds_left, np.abs(state.delta_num_rounds_won),
              state.delta_num_rounds_won, -state.num_coins_left_player1,
              -state.num_coins_left_player2)

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

  def serialize(self):
    '''Returns a serialized version of this strategy book.

    Returns a string.
    '''
    strategy_book_proto = strategy_pb2.StrategyBook()
    for state, (strategy, value) in self._strategies.items():
      state_data = strategy_book_proto.states.add()
      state_data.state.num_coins_left_player1 = state.num_coins_left_player1
      state_data.state.num_coins_left_player2 = state.num_coins_left_player2
      state_data.state.delta_num_rounds_won = state.delta_num_rounds_won
      state_data.state.num_rounds_left = state.num_rounds_left

      for probability in strategy:
        state_data.strategy.action_probabilities.append(probability)

      state_data.value = value
    return strategy_book_proto.SerializeToString()

  def load_from_serialized(self, serialized):
    '''Loads strategies from a serialized strategy book.'''
    strategy_book_proto = strategy_pb2.StrategyBook()
    strategy_book_proto.ParseFromString(serialized)
    for state_data in strategy_book_proto.states:
      state = cyclops.State(state_data.state.num_coins_left_player1,
                            state_data.state.num_coins_left_player2,
                            state_data.state.delta_num_rounds_won,
                            state_data.state.num_rounds_left)

      strategy = [p for p in state_data.strategy.action_probabilities]
      self._strategies[state] = (strategy, state_data.value)
