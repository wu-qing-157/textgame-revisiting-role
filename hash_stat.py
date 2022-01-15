from env import JerichoEnv
from argparse import Namespace


def stat_state_hash(rom, t='gt_state'):
    if t == 'gt_state':
        env = JerichoEnv(rom, Namespace(nor=False, randr=False, perturb=False, use_gt_state=True, use_gt_room=False, use_nearby=0, output_dir='logs_temp', no_current=False, no_last_look=False))
    elif t == 'gt_room':
        env = JerichoEnv(rom, Namespace(nor=False, randr=False, perturb=False, use_gt_state=False, use_gt_room=True, use_nearby=0, output_dir='logs_temp', no_current=False, no_last_look=False))
    elif t == 'nearby':
        env = JerichoEnv(rom, Namespace(nor=False, randr=False, perturb=False, use_gt_state=True, use_gt_room=False, use_nearby=1, output_dir='logs_temp', no_current=False, no_last_look=False))
    else:
        assert False
    walkthrough = env.env.get_walkthrough()
    ret = []
    _, info = env.reset()
    ret.append(hash(info['state_hash']))
    for action in walkthrough:
        _, _, done, info = env.step(action)
        ret.append(hash(info['state_hash']))
        if done:
            break
    count = 0
    for i, j in zip(ret[:-1], ret[1:]):
        if i != j:
            count += 1
    print(len(ret), count)


if __name__ == '__main__':
    stat_state_hash('zork1.z5', t='gt_state')
    stat_state_hash('zork1.z5', t='gt_room')
    stat_state_hash('zork1.z5', t='gt_nearby')
