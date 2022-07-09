
import launchpad as lp

from absl import app
from absl import logging

import time
import copy
import numpy as np

RUNNERS = 100
MAX_RANGE = 100

class Counter:
    def __init__(self) -> None:
        self.count = 0
        self.dataset = []
    def increment(self, value=1):
        self.count += 1
    def get_count(self):
        return self.count
    def add_batch_data(self, list_data):
        name, arr = list_data
        for i in arr:
            self.dataset.append(name+str(i))
        # return np.array(0)
    def get_dataset(self):
        return self.dataset


class Runner:
    def __init__(self, counter, name) -> None:
        self.counter = counter
        self.name = name
    def run(self):
        add_name = lambda n: self.name+str(n)
        for i in range(0,MAX_RANGE,5):
            self.counter.add_batch_data((self.name, np.arange(i)))
            self.counter.increment()
            time.sleep(0.5)


class Monitor:
    def __init__(self, counter) -> None:
        self.counter = counter
    def run(self):
        while self.counter.get_count()<(RUNNERS*MAX_RANGE)//5:
            print(self.counter.get_count())
            time.sleep(1)
        print(self.counter.get_count())
        with open("data.txt", "w") as f:
            f.write("\n".join(self.counter.get_dataset()))
        # print(self.counter.get_dataset())
        lp.stop()


def make_program():
    program = lp.Program('program_wait')
    # counter = program.add_node(lp.CourierNode(Counter), label='counter_node')
    counter = Counter()
    for i in range(RUNNERS):
        program.add_node(lp.CourierNode(Runner, counter, str(i)+"r"), label='runner'+str(i))
    program.add_node(lp.CourierNode(Monitor, counter), label='monitor_node')
    return program


def main(argv):
  start = time.time()
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  program = make_program()
  controller = lp.launch(program)
  if not controller:
    logging.info('Waiting for program termination is not supported.')
    return
  controller.wait()
  logging.info('Program finished.')
  print(f"Finished in {time.time()-start}")

if __name__ == '__main__':
  app.run(main)

# Conclusion: 
# This program shows that the objects passed to each LP node is shared among all.
# This may slowdown the program execution and may cause unpredictable behavior.
