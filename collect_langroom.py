import argparse
import collections
import concurrent.futures
import math
import os
import random

import langroom
import numpy as np
from PIL import Image
from tqdm import tqdm


def _save_episode(dataset_path, split, split_info, episode_id, episode_observations, episode_actions, episode_rewards):
    split_path = os.path.join(dataset_path, split)
    os.makedirs(split_path, exist_ok=True)
    np.save(os.path.join(split_path, 'info.npy'), split_info, allow_pickle=True)

    episode_info = split_info[episode_id]
    actions_path = os.path.join(split_path, episode_info['actions'])
    episode_path = os.path.dirname(actions_path)
    os.makedirs(episode_path, exist_ok=True)
    np.save(actions_path, np.asarray(episode_actions), allow_pickle=True)

    rewards_path = os.path.join(split_path, episode_info['rewards'])
    np.save(rewards_path, np.asarray(episode_rewards, dtype=np.float32), allow_pickle=True)

    for obs_relative_path, observation in zip(split_info[episode_id]['obs'], episode_observations):
        obs_path = os.path.join(split_path, obs_relative_path)
        Image.fromarray(observation).resize(size=(96, 96), resample=Image.Resampling.NEAREST).save(obs_path)


def save_episode(dataset_path, val_fraction, counter, executor, futures, pbar, max_futures, episode_observations, episode_actions, episode_rewards,):
    split = 'val' if random.random() <= val_fraction else 'train'
    split_info = {}
    episode_id = counter[split]
    counter[split] += 1

    episode_folder_path = f'obs/ep_{episode_id}'
    observation_paths = [f'{episode_folder_path}/s_{i}.JPEG' for i in range(len(episode_observations))]
    actions_path = f'{episode_folder_path}/actions.npy'
    rewards_path = f'{episode_folder_path}/rewards.npy'
    episode_info = {'obs': observation_paths, 'actions': actions_path, 'rewards': rewards_path}
    split_info['num_episodes'] = counter[split]
    split_info[episode_id] = episode_info

    while (len(futures) > 0 and futures[0].done()) or len(futures) > max_futures:
        futures.popleft().result()
        pbar.update(1)

    future = executor.submit(_save_episode, dataset_path, split, split_info, episode_id,
                                  episode_observations, episode_actions, episode_rewards)
    futures.append(future)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--train_episodes', type=int, required=True)
    parser.add_argument('--val_episodes', type=int, required=True)
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--max_episode_length', type=int, default=math.inf)
    args = parser.parse_args()

    val_fraction = args.val_episodes / (args.train_episodes + args.val_episodes)
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers)
    futures = collections.deque()
    max_futures = 2 * args.num_workers
    pbar = tqdm(total=args.train_episodes + args.val_episodes, position=tqdm._get_free_pos(), desc='# Saved episodes')


    env = langroom.LangRoom(resolution=94)
    counter = collections.Counter({'train': 0, 'val': 0})

    while sum(counter.values()) < args.train_episodes + args.val_episodes:
        action = {'reset': True}
        observations = [env.step(action)['image']]
        actions = []
        rewards = []
        done = False
        while not done and len(observations) < args.max_episode_length:
            action = {k: v.sample() for k, v in env.act_space.items()}
            action['reset'] = False
            obs = env.step(action)
            action.pop('reset')
            actions.append(action)
            observations.append(obs['image'])
            rewards.append(obs['reward'])
            done = obs['is_last'] or obs['is_terminal']

        save_episode(args.dataset_path, val_fraction, counter, executor, futures, pbar, max_futures, observations, actions, rewards)

    for future in futures:
        future.result()
        pbar.update(1)

    pbar.close()
    executor.shutdown()
