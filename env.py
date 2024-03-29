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
        self.last_look = {}
        self.use_gt_state_hash = args.use_gt_state
        self.use_gt_room = args.use_gt_room
        self.use_nearby_room = args.use_nearby_room
        self.log_dir = args.output_dir
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

    @staticmethod
    def get_room(look):
        room = look.split('\n')[0].strip()
        if '.' in room or not room:
            room = 'unknown'
        return room

    def get_nearby(self, depth=1):
        state = self.env.get_state()
        valid = self.env.get_valid_actions(state, use_ctypes=False)
        navis = 'north/south/west/east/northwest/southwest/northeast/southeast/up/down'.split('/')
        result = []
        for act in navis:  # ensure the order of acts in different states
            if act in valid:
                self.env.step(act)
                next_state = self.env.get_state()
                look, _, _, _ = self.env.step('look')
                self.env.set_state(next_state)
                room = self.get_room(look.lower())
                if depth > 1 and room != 'unknown':
                    result.append((act, room, self.get_nearby(depth - 1)))
                else:
                    result.append((act, room))
                self.env.set_state(state)
        return tuple(result)

    def last_look_hash(self, location, inventory):
        return hash(tuple(sorted(self.last_look.items())) + (location, inventory))

    def get_state_hash(self, look, inv):
        if self.use_gt_state_hash:
            return self.env.get_world_state_hash()
        else:
            if self.use_gt_room:
                location = int(self.env.get_player_location().num)
            elif self.use_nearby_room:
                location = (self.get_room(look), self.get_nearby(depth=self.use_nearby_room))
                with open(f'{self.log_dir}/location.log', 'a') as f:
                    print(int(self.env.get_player_location().num), file=f)
                    print(repr(location), file=f)
            else:
                location = self.get_room(look)
            self.last_look[location] = hash(look)
        return self.last_look_hash(location, inv)

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
                    valid = self.env.get_valid_actions(use_ctypes=False)
                    if len(valid) == 0:
                        valid = ['wait', 'yes', 'no']
                    info['valid'] = valid
                if self.cache is not None:
                    self.cache[hash_save] = info['look'], info['inv'], info['valid']

        self.steps += 1
        if self.step_limit and self.steps >= self.step_limit:
            done = True
        self.max_score = max(self.max_score, info['score'])
        if done: self.end_scores.append(info['score'])
        if self.perturb: 
            ob = self.paraphrase(ob)
            info['look'] = self.paraphrase(info['look'])
            info['inv'] = self.paraphrase(info['inv'])
        info['state_hash'] = self.get_state_hash(info['look'], info['inv'])
        return ob, reward, done, info

    def reset(self):
        initial_ob, info = self.env.reset()
        save = self.env.get_state()
        self.walkthrough = deque(self.env.get_walkthrough())
        look, _, _, _ = self.env.step('look')
        info['look'] = look.lower()
        self.env.set_state(save)
        inv, _, _, _ = self.env.step('inventory')
        info['inv'] = inv.lower()
        self.env.set_state(save)
        valid = self.env.get_valid_actions(use_ctypes=False)
        info['valid'] = valid
        self.steps = 0
        self.max_score = 0
        self.objs = set()
        self.last_look.clear()
        info['state_hash'] = self.get_state_hash(info['look'], info['inv'])
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
