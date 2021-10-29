from jericho import *
from jericho.util import *
from jericho.defines import *
import numpy as np
import torch
from collections import deque, defaultdict
import json


def modify(info):
    info['look'] = clean(info['look'])

    # shuffle inventory
    info['inv'] = clean(info['inv'])
    invs = info['inv'].split('  ')
    if len(invs) > 1:
        head, invs = invs[0], invs[1:]
        np.random.shuffle(invs)
        info['inv'] = '  '.join([head] + invs)

    # action switch
    subs = [['take', 'grab'], ['drop', 'put down'], ['turn on', 'open'], ['turn off', 'close'], ['get in', 'enter']]
    navis = 'north/south/west/east/northwest/southwest/northeast/southeast/up/down'.split('/')
    navi_subs = [[w, 'go ' + w] for w in navis]
    def sub(act):
        for a, b in subs:
            act = act.replace(a, b)
        for a, b in navi_subs:
            if act == a: act = b
        return act
    info['valid'] = [sub(act) for act in info['valid']]

    return info

class JerichoEnv:
    ''' Returns valid actions at each step of the game. '''

    def __init__(self, rom_path, seed, step_limit=None, get_valid=True, cache=None, args=None):
        self.rom_path = rom_path
        self.env = FrotzEnv(rom_path, seed=seed)
        self.bindings = self.env.bindings
        self.seed = seed
        self.steps = 0
        self.step_limit = step_limit
        self.get_valid = get_valid
        self.max_score = 0
        self.end_scores = []
        self.cache = cache
        self.nor = args.nor
        self.randr = args.randr
        np.random.seed(max(seed, 0))
        self.random_rewards = (np.random.rand(10000) - .5) * 10.        
        self.objs = set()
        self.perturb = args.perturb
        if self.perturb:
            self.en2de = args.en2de 
            self.de2en = args.de2en 
            self.perturb_dict = args.perturb_dict
        # self.ram_bytes = defaultdict(lambda: set())
        # self.stack_bytes = defaultdict(lambda: set())
    
    def paraphrase(self, s):
        if s in self.perturb_dict: return self.perturb_dict[s]
        with torch.no_grad():
            p = self.de2en.translate(self.en2de.translate(s))
        if p == '': p = '.'
        self.perturb_dict[s] = p
        return p

    def get_objects(self):
        desc2objs = self.env._identify_interactive_objects(use_object_tree=False)
        obj_set = set()
        for objs in desc2objs.values():
            for obj, pos, source in objs:
                if pos == 'ADJ': continue
                obj_set.add(obj)
        return list(obj_set)

    def step(self, action):
        # if not self.walkthrough:
        #     with open('ram_bytes.json', 'w') as f:
        #         for i, l in sorted((int(i), sorted(map(int, j))) for i, j in self.ram_bytes.items()):
        #             print(str(i), ':', ' '.join(map(str, l)), file=f)
        # action = self.walkthrough.popleft()
        ob, reward, done, info = self.env.step(action)
        # if self.cache is not None:
        #     self.cache['loc'].add(self.env.get_player_location().num)
        if self.nor: reward = 0
        # random reward
        if self.randr:  
            reward = 0
            objs = [self.env.get_player_location()] + self.env.get_inventory()
            for obj in objs:
                obj = obj.num
                if obj not in self.objs:
                    self.objs.add(obj)
                    reward += self.random_rewards[obj]
            info['score'] = sum(self.random_rewards[obj] for obj in self.objs)

        # Initialize with default values
        info['look'] = 'unknown'
        info['inv'] = 'unknown'
        info['valid'] = ['wait', 'yes', 'no']
        if not done:
            save = self.env.get_state()
            # for i in np.nonzero(self.last_ram != save[0])[0]:
            #     self.ram_bytes[i].update((self.last_ram[i], save[0][i]))
            # # print(self.env.get_score(), len(self.ram_bytes), sorted(self.ram_bytes), sorted(self.stack_bytes), save[1][0])
            # for i in range(len(self.env.get_world_objects())):
            #     if tuple(self.properties[i]) != tuple(self.env.get_world_objects()[i].properties):
            #         print(''.join(str(p) for p in self.properties[i]))
            #         print(''.join(str(p) for p in self.env.get_world_objects()[i].properties))
            #         self.properties[i] = self.env.get_world_objects()[i].properties
            # for i in range(len(self.env.get_world_objects())):
            #     if np.nonzero(self.attributes[i])[0].tolist() != np.nonzero(self.env.get_world_objects()[i].attr)[0].tolist():
            #         print(self.env.get_world_objects()[i].name)
            #         print(np.nonzero(self.attributes[i])[0].tolist(), np.nonzero(self.env.get_world_objects()[i].attr)[0].tolist())
            #         self.attributes[i] = self.env.get_world_objects()[i].attr
            # with open('object.txt', 'a') as f:
            #     for i in self.env.get_world_objects():
            #         print(i, file=f)
            #     print('====', file=f)
            # self.last_ram = save[0]
            # self.last_stack = save[1]
            hash_save = self.env.get_world_state_hash() 
            if self.cache is not None and hash_save in self.cache:
                info['look'], info['inv'], info['valid'] = self.cache[hash_save]
            else:
                look, _, _, _ = self.env.step('look')
                info['look'] = look.lower()
                self.env.set_state(save)
                inv, _, _, _ = self.env.step('inventory')
                info['inv'] = inv.lower()
                self.env.set_state(save)
                if self.get_valid:
                    valid = self.env.get_valid_actions()
                    if len(valid) == 0:
                        valid = ['wait', 'yes', 'no']
                    info['valid'] = valid
                if self.cache is not None:
                    self.cache[hash_save] = info['look'], info['inv'], info['valid'] 
        # with open('traj.txt', 'a') as f:
        #     print(f'Step {self.steps} | Score {self.env.get_score()}', file=f)
        #     print(ob, file=f, end='')
        #     print(info['look'], file=f, end='')
        #     print(info['inv'], file=f, end='')
        #     print('====', file=f)

        location = info['look'].split('\n')[0]
        if location in ['forest', 'clearing']:
            location = info['look'].split('\n')[1]
        self.last_look[location] = hash(info['look'])

        self.steps += 1
        if self.step_limit and self.steps >= self.step_limit:
            done = True
        self.max_score = max(self.max_score, info['score'])
        if done: self.end_scores.append(info['score'])
        if self.perturb: 
            ob = self.paraphrase(ob)
            info['look'] = self.paraphrase(info['look'])
            info['inv'] = self.paraphrase(info['inv'])
        info['state_hash'] = self.last_look_hash() #self.env.get_world_state_hash()
        return ob, reward, done, info
    
    def last_look_hash(self):
        # print(tuple(v for _, v in sorted(self.last_look.items())))
        return hash(tuple(v for _, v in sorted(self.last_look.items())))

    def reset(self):
        initial_ob, info = self.env.reset()
        save = self.env.get_state()
        # self.last_ram = save[0]
        # self.last_stack = save[1]
        self.walkthrough = deque(self.env.get_walkthrough())
        # self.properties = [i.properties for i in self.env.get_world_objects()]
        # self.attributes = [i.attr for i in self.env.get_world_objects()]
        look, _, _, _ = self.env.step('look')
        info['look'] = look
        self.env.set_state(save)
        inv, _, _, _ = self.env.step('inventory')
        info['inv'] = inv
        self.env.set_state(save)
        valid = self.env.get_valid_actions()
        info['valid'] = valid
        self.steps = 0
        self.max_score = 0
        self.objs = set()
        self.last_look = {}
        location = info['look'].split('\n')[0]
        if location in ['forest', 'clearing']:
            location = info['look'].split('\n')[1]
        self.last_look[location] = hash(info['look'])
        info['state_hash'] = self.last_look_hash() #self.env.get_world_state_hash()
        return initial_ob, info

    def get_dictionary(self):
        if not self.env:
            self.create()
        return self.env.get_dictionary()

    def get_action_set(self):
        return None

    def get_end_scores(self, last=1):
        last = min(last, len(self.end_scores))
        return sum(self.end_scores[-last:]) / last if last else 0

    def close(self):
        self.env.close()
