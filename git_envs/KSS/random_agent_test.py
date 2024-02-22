import random
from KSS_env_framework import KSSFramework

class RandomAgent:
    def __init__(self, action_num):
        self.action_num = action_num

    def initialize(self, exercises_record):
        pass

    def take_action(self):
        return random.randint(0, self.action_num-1)

    def refresh(self, action, correct):
        return


env = KSSFramework()
agent = RandomAgent(action_num=10)

records = env.test(agent=agent)
print(len(records), records)

