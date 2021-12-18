import os

run_types = [
    ('--use_gt_state', 'gt_state'),
    ('--use_gt_room', 'gt_room'),
    ('--use_nearby_room=2', 'nearby'),
    ('', 'look')
]

cases = []
cuda = 0
for game in ['zork1', 'inhumane', 'ludicorp', 'pentari', 'detective', 'balances', 'deephome', 'dragon']:
    for run_arg, run_name in run_types:
        cases.append((cuda, game, run_arg, run_name))
        cuda = (cuda + 1) % 4

roms = os.listdir('roms')
for cuda, game, run_arg, run_name in cases:
    for rom in roms:
        if game in rom:
            rom_path = f'roms/{rom}'
            break
    else:
        assert False, f'cannot find rom {game}'
    command = f'CUDA_VISIBLE_DEVICES={cuda} python train.py --rom_path={rom_path} --w_inv=1 --w_act=1 --r_for=1 {run_arg} --output_dir=logs_inv/{game}-{run_name}'
    print(command)
