import argparse
import numpy as np

from strategy_book import StrategyBook
import cyclops


def prompt_num_coins(max_num_coins):
  '''Prompts the user for how many coins to spend.

  Returns an integer.
  '''
  while True:
    result = input('How many coins? ')
    try:
      result = int(result)
    except:
      print('You must enter a number. Try again.')
      continue
    if result < 0 or result > max_num_coins:
      print('Enter a number between 0 and {}'.format(max_num_coins))
      continue
    return result


def pretty_print_state(state, num_rounds_won_player1, num_rounds_won_player2):
  return ('Computer: {} coins ({} rounds won). '
          'You: {} coins ({} rounds won). {} rounds remaining'.format(
              state.num_coins_left_player1, num_rounds_won_player1,
              state.num_coins_left_player2, num_rounds_won_player2,
              state.num_rounds_left))


def main():
  parser = argparse.ArgumentParser(description='Plays a game.')
  parser.add_argument(
      '--strategy_file_in',
      type=str,
      help='path to the input file for the strategy book')
  parser.add_argument(
      '--play_before_computer',
      action='store_true',
      help=('if true, you have to enter the number of coins you spend before '
            'you can see what the computer spends'))
  args = parser.parse_args()

  input_file = args.strategy_file_in
  with open(input_file, 'rb') as f:
    serialized_strategy_book = f.read()

  strategy_book = StrategyBook()
  strategy_book.load_from_serialized(serialized_strategy_book)

  state = cyclops.State.initial_state()

  num_rounds_won_player1 = 0
  num_rounds_won_player2 = 0

  while not state.is_terminal():
    print('')
    print(pretty_print_state(state, num_rounds_won_player1,
                             num_rounds_won_player2))
    actions_player2 = state.get_actions_player2()

    strategy = strategy_book.get_strategy(state)
    if not strategy:
      raise Exception('No recorded strategy for state {}'.format(state))

    strategy = np.array(strategy)
    assert (abs(1 - np.sum(strategy)) < 0.00001)
    strategy /= np.sum(strategy)

    num_coins_spent_player1 = np.random.choice(
        state.get_actions_player1(), p=strategy)

    num_coins_spent_player2 = None
    if len(state.get_actions_player2()) == 1:
      num_coins_spent_player2 = state.get_actions_player2()[0]
    if num_coins_spent_player2 is None and args.play_before_computer:
      num_coins_spent_player2 = prompt_num_coins(np.max(actions_player2))

    print('Computer is spending... {} coins.'.format(num_coins_spent_player1))

    if num_coins_spent_player2 is None and not args.play_before_computer:
      num_coins_spent_player2 = prompt_num_coins(np.max(actions_player2))

    num_rounds_won_player1 += num_coins_spent_player1 > num_coins_spent_player2
    num_rounds_won_player2 += num_coins_spent_player2 > num_coins_spent_player1

    state = state.next_state(num_coins_spent_player1, num_coins_spent_player2)

  print('')
  score = state.score()
  if score > 0:
    print('You lose!')
  elif score == 0:
    print('The game is tied. Well done!')
  else:
    print('I reluctantly admit that you won. You must be some kind of genius.')


if __name__ == '__main__':
  main()
