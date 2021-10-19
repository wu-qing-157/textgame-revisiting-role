import jericho
import pickle

env = jericho.FrotzEnv('zork1.z5')
with open('state.pickle', 'rb') as f:
    state = pickle.load(f)
env.set_state(state)

env.step('put sack in egg')
print('here')
print(env.get_valid_actions())
