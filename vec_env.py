import numpy as np
import os
from multiprocessing import Process, Pipe

def worker(remote, parent_remote, env, log_file):
    parent_remote.close()
    log_file_ = open(log_file, 'w')
    try:
        done = False
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                print(f'step {data}', end='', file=log_file_)
                log_file_.flush()
                if done:
                    ob, info = env.reset()
                    reward = 0
                    done = False
                else:
                    ob, reward, done, info = env.step(data)
                remote.send((ob, reward, done, info))
            elif cmd == 'reset':
                print(f'reset', end='', file=log_file_)
                log_file_.flush()
                ob, info = env.reset()
                remote.send((ob, info))
            elif cmd == 'get_end_scores':
                print(f'get_end_scores', end='', file=log_file_)
                log_file_.flush()
                remote.send(env.get_end_scores(last=100))
            elif cmd == 'close':
                print(f'close', end='', file=log_file_)
                log_file_.flush()
                env.close()
                break
            else:
                raise NotImplementedError
            print(f'|', file=log_file_)
            log_file_.flush()
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        log_file_.close()
        env.close()


class VecEnv:
    def __init__(self, num_envs, env, log_dir):
        self.closed = False
        self.num_envs = num_envs
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(num_envs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, env, os.path.join(log_dir, f'env_{idx}.log')))
                   for (work_remote, remote, idx) in zip(self.work_remotes, self.remotes, range(num_envs))]
        for p in self.ps:
            # p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def __del__(self):
        for p in self.ps:
            p.kill()

    def step(self, actions):
        self._assert_not_closed()
        assert len(actions) == self.num_envs, "Error: incorrect number of actions."
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rewards, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rewards), np.stack(dones), infos

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        obs, infos = zip(*results)
        return np.stack(obs), infos
    
    def get_end_scores(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('get_end_scores', None))
        results = [remote.recv() for remote in self.remotes]
        return np.stack(results)

    def close_extras(self):
        self.closed = True
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"
