import os

# games = ['zork1', 'inhumane', 'ludicorp', 'pentari', 'detective', 'balances', 'deephome', 'dragon']
games = ['zork1', 'zork3', 'pentari', 'ludicorp', 'inhumane', 'detective', 'balances', 'omniquest', 'library']
run_types = [
    # ('--use_q_att --use_inv_att --use_gt_state', 'att-gt_state'),
    # ('--use_q_att --use_inv_att --use_gt_room', 'att-gt_room'),
    # ('--use_gt_state', 'gt_state'),
    # ('--use_gt_room', 'gt_room'),
    ('--use_q_att --use_inv_att --use_nearby_room=2', 'nearby'),
    ('--use_nearby_room=2', 'nearby'),
    # ('', 'look'),
    # ('', 'raw'),
    # ('--use_gt_state --hash_only', 'gt_state-hash_only'),
]
seeds = [3]

cases = []
cuda = 0
for game in games:
    for run_arg, run_name in run_types:
        for seed in seeds:
            cases.append((cuda, game, run_arg, run_name, seed))
            cuda = (cuda + 1) % 4

roms = os.listdir('roms')
for cuda, game, run_arg, run_name, seed in cases:
    for rom in roms:
        if game in rom:
            rom_path = f'roms/{rom}'
            break
    else:
        assert False, f'cannot find rom {game}'
    command = f'CUDA_VISIBLE_DEVICES={cuda} python train.py --rom_path={rom_path} --w_inv=1 --w_act=1 --r_for=1 --seed={seed} {run_arg} --output_dir=logs_mlp/{game}-{run_name}-{seed}'
    print(command)
