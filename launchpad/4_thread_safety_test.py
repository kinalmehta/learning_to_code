
import launchpad as lp

from absl import app
from absl import logging

import time


class Counter:
    def __init__(self) -> None:
        self.count = 0
    def increment(self, value=1):
        self.count += 1
    def get_count(self):
        return self.count


class Runner:
    def __init__(self, counter) -> None:
        self.counter = counter
    def run(self):
        for i in range(100000):
            self.counter.increment()


class Monitor:
    def __init__(self, counter) -> None:
        self.counter = counter
    def run(self):
        while self.counter.get_count()<200000:
            print(self.counter.get_count())
            # time.sleep(1)
        print(self.counter.get_count())
        lp.stop()

def make_program():
    program = lp.Program('program_wait')
    counter = program.add_node(lp.CourierNode(Counter), label='counter_node')
    program.add_node(lp.CourierNode(Runner, counter), label='runner1')
    program.add_node(lp.CourierNode(Runner, counter), label='runner2')
    program.add_node(lp.CourierNode(Monitor, counter), label='monitor_node')
    return program


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  program = make_program()
  controller = lp.launch(program)
  if not controller:
    logging.info('Waiting for program termination is not supported.')
    return
  controller.wait()
  logging.info('Program finished.')

if __name__ == '__main__':
  app.run(main)

