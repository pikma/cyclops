STRATEGY_FILE=/tmp/strategy.pb

play: $(STRATEGY_FILE) play.py
	pipenv run python play.py --strategy_file_in=$(STRATEGY_FILE) --play_before_computer

ai: $(STRATEGY_FILE) play.py
	pipenv run python play.py --strategy_file_in=$(STRATEGY_FILE)

clean_strategy:
	rm $(STRATEGY_FILE)

prepare: $(STRATEGY_FILE)

$(STRATEGY_FILE): *.py
	pipenv run python prepare.py --strategy_file_out=$(STRATEGY_FILE)

strategy_pb2.py: strategy.proto
	protoc --python_out=. strategy.proto

clean:
	rm -f strategy_pb2.py
	yapf -i -r .

test:
	pipenv run python main_test.py
