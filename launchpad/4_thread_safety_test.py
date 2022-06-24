
import launchpad as lp

from absl import app
from absl import logging

import time


class Counter:
    def __init__(self) -> None:
        self.count = 0
        self.dataset = []
    def increment(self, value=1):
        self.count += 1
    def get_count(self):
        return self.count
    def add_batch_data(self, list_data):
        for i in list_data:
            self.dataset.append(i)
    def get_dataset(self):
        return self.dataset


class Runner:
    def __init__(self, counter, name) -> None:
        self.counter = counter
        self.name = name
    def run(self):
        add_name = lambda n: self.name+str(n)
        for i in range(0,1000,5):
            self.counter.add_batch_data(list(map(add_name,list(range(i)))))
            self.counter.increment()


class Monitor:
    def __init__(self, counter) -> None:
        self.counter = counter
    def run(self):
        while self.counter.get_count()<2000//5:
            print(self.counter.get_count())
            time.sleep(1)
        print(self.counter.get_count())
        with open("data.txt", "w") as f:
            f.write("\n".join(self.counter.get_dataset()))
        # print(self.counter.get_dataset())
        lp.stop()

def make_program():
    program = lp.Program('program_wait')
    counter = program.add_node(lp.CourierNode(Counter), label='counter_node')
    program.add_node(lp.CourierNode(Runner, counter, "1r"), label='runner1')
    program.add_node(lp.CourierNode(Runner, counter, "2r"), label='runner2')
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

