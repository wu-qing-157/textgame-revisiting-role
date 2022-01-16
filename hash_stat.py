import sys

from env import JerichoEnv
from argparse import Namespace
from tqdm import tqdm


def stat_state_hash(rom, traj, t='gt_state', log_same=False):
    if t == 'gt_state':
        env = JerichoEnv(rom, 0, get_valid=False, args=Namespace(nor=False, randr=False, perturb=False, use_gt_state=True, use_gt_room=False, use_nearby_room=0, output_dir='logs_temp', no_current=False, no_last_look=False))
    elif t == 'gt_room':
        env = JerichoEnv(rom, 0, get_valid=False, args=Namespace(nor=False, randr=False, perturb=False, use_gt_state=False, use_gt_room=True, use_nearby_room=0, output_dir='logs_temp', no_current=False, no_last_look=False))
    elif t == 'nearby':
        env = JerichoEnv(rom, 0, get_valid=False, args=Namespace(nor=False, randr=False, perturb=False, use_gt_state=False, use_gt_room=False, use_nearby_room=1, output_dir='logs_temp', no_current=False, no_last_look=False))
    else:
        env = JerichoEnv(rom, 0, get_valid=False, args=Namespace(nor=False, randr=False, perturb=False, use_gt_state=False, use_gt_room=False, use_nearby_room=0, output_dir='logs_temp', no_current=False, no_last_look=False))
    # walkthrough = env.env.get_walkthrough()
    ret = set()

    def update(obs_, info_):
        if t == 'drrn_hash':
            ret.add((hash((obs_, info_['look'], info_['inv']))))
        else:
            ret.add(hash(info_['state_hash']))

    obs, info = env.reset()
    update(obs, info)
    for line in tqdm(open(traj).readlines()):
        if line.startswith('>> Action'):
            action = line.split(':')[1][1:]
            obs, _, _, info = env.step(action)
            update(obs, info)
        elif line.startswith('Starting evaluation episode'):
            obs, info = env.reset()
            update(obs, info)
        elif line.startswith('Action'):
            action = line.split(':')[1][1:].split(',')[0]
            obs, _, _, info = env.step(action)
            update(obs, info)
    print(t, len(ret))


if __name__ == '__main__':
    rom = sys.argv[1]
    traj = sys.argv[2]
    stat_state_hash(rom, traj, t='gt_state', log_same=True)
    stat_state_hash(rom, traj, t='gt_room', log_same=True)
    stat_state_hash(rom, traj, t='nearby')
    stat_state_hash(rom, traj, t='room_name')
    stat_state_hash(rom, traj, t='drrn')
