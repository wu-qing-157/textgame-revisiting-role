import sys

from env import JerichoEnv
from argparse import Namespace
from tqdm import tqdm


def stat_state_hash(rom, t='gt_state', log_same=True):
    if t == 'gt_state':
        env = JerichoEnv(rom, 0, get_valid=False, args=Namespace(nor=False, randr=False, perturb=False, use_gt_state=True, use_gt_room=False, use_nearby_room=0, output_dir='logs_temp', no_current=False, no_last_look=False))
    elif t == 'gt_room':
        env = JerichoEnv(rom, 0, get_valid=False, args=Namespace(nor=False, randr=False, perturb=False, use_gt_state=False, use_gt_room=True, use_nearby_room=0, output_dir='logs_temp', no_current=False, no_last_look=False))
    elif t == 'nearby':
        env = JerichoEnv(rom, 0, get_valid=False, args=Namespace(nor=False, randr=False, perturb=False, use_gt_state=True, use_gt_room=False, use_nearby_room=1, output_dir='logs_temp', no_current=False, no_last_look=False))
    else:
        env = JerichoEnv(rom, 0, get_valid=False, args=Namespace(nor=False, randr=False, perturb=False, use_gt_state=True, use_gt_room=False, use_nearby_room=0, output_dir='logs_temp', no_current=False, no_last_look=False))
    walkthrough = env.env.get_walkthrough()
    ret = []
    obs, info = env.reset()
    if t == 'drrn_hash':
        ret.append((hash((obs, info['look'], info['inv']))))
    else:
        ret.append(hash(info['state_hash']))
    count = 0
    for action in tqdm(walkthrough):
        obs, _, done, info = env.step(action)
        if t == 'drrn_hash':
            ret.append((hash((obs, info['look'], info['inv']))))
        else:
            ret.append(hash(info['state_hash']))
        if ret[-1] != ret[-2]:
            count += 1
        elif log_same:
            print(action)
            print(obs.strip())
            print(info['look'].strip())
        if done:
            break
    print(env.env.get_score())
    print(t, len(ret) - 1, count)


if __name__ == '__main__':
    rom = sys.argv[1]
    stat_state_hash(rom, t='gt_state')
    stat_state_hash(rom, t='gt_room')
    stat_state_hash(rom, t='nearby')
    stat_state_hash(rom, t='room_name')
    stat_state_hash(rom, t='drrn')
