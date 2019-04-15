import numpy as np
import tensorflow as tf
import gym
from collections import namedtuple
import itertools
import time
from pnet_softmax import PolicyNet
from sklearn.preprocessing import minmax_scale, scale
import gym_dlr


normalize_metric_wise = False
train_round = 1
fixed_actions = False  # True when testing with fixed actions


restore_ckpt = './checkpoints/pg_agent_{}.ckpt'.format(train_round-1)
save_ckpt = './checkpoints/pg_agent_{}.ckpt'.format(train_round)

resume = True
sequential = True  # sequential sampling not random
baseline = False  # no action will be taken if true
raw_reward = False  # True when testing and baseline, no training

epsilon = 1e-7

EpisodeStats = namedtuple("stats", ["obs", "acts", "cos_values", "rewards"])
env_name = 'Dlr-v0'

SAR_NODE_NAMES = ['ceph-2', 'ceph-3', 'ceph-4', 'ceph-5', 'ceph-6', 'ceph-7', 'ceph-8']
NUM_NODES = len(SAR_NODE_NAMES)  # cluster size excluding master node


TOTAL_STATS = 7  # [io_wait_array, net_usage_array, rtps_array, wtps_array, cpu_usage_array, affinity_array, reweight_array]
USEFUL_STATS = 4  # [io_wait_array, net_usage_array, rtps_array, wtps_array] # <---should be in same order

STAT_NAMES = ['io_wait', 'net_usage', 'rtps', 'wtps', 'cpu_usage', 'affinity', 'weight']


OBS_SHAPE = (USEFUL_STATS * NUM_NODES, )
# TODO: this list should contain action functions
# ACTIONS = ['cpu_a', 'net_a', 'io_a', 'cpu_w',  'io_w']
ACTIONS = ['io_a', 'net_a', 'io_w', 'net_w', 'nop'] # <---should be in same order
N_ACTIONS = len(ACTIONS)

# Network Related
fc1_units = 180
gamma = 0.99  # discount factor for reward
batch_size = 6  # after how many episodes to do a policy update?
n_episodes = 900
LEARNING_RATE = 0.001


# TODO: check if some pre processing needed
def get_state(curr_obs, prev_obs=None):
    """Convert observations to useful states."""
    curr_obs = curr_obs[:USEFUL_STATS]  # (4, 7) # (rows=stats, cols=nodes)
    if curr_obs.shape == (USEFUL_STATS, NUM_NODES):
        state = curr_obs  # - prev_obs if prev_obs is not None else np.zeros(OBS_SHAPE)
        if normalize_metric_wise:
            state = scale(state)
        return state.ravel()
    else:
        print("******************************")
        print("ERROR: OBSERVATION SHAPE MISMATCH, got {}, needed {}".format(curr_obs.shape, (USEFUL_STATS, NUM_NODES)))
        print("OBSERVATIONS: ", curr_obs)
        print("******************************")
        return None


def get_new_values(curr_obs, action_idx, action):
    index = action_idx % 2
    chosen_usage = curr_obs[index]
    chosen_usage_mean = np.mean(chosen_usage)
    deltas = np.array([(u - chosen_usage_mean)/(u + epsilon) for u in chosen_usage])

    if action[-1] == 'w':
        weights = curr_obs[-1]
        deltas = np.clip(deltas, a_min=-0.4, a_max=0.4)
        new_weights = np.concatenate((np.clip((weights - deltas), a_min=0.1, a_max=1.), [0.]), axis=0)  # add flag
        assert new_weights.shape[0] == NUM_NODES + 1
        return new_weights

    elif action[-1] == 'a':
        affinities = curr_obs[-2]
        new_affinities = np.concatenate((np.clip((affinities - 2 * deltas), a_min=0., a_max=1.), [1.]), axis=0)  # add flag
        assert new_affinities.shape[0] == NUM_NODES + 1
        return new_affinities

    elif action == 'nop':
        old_affinities = curr_obs[-2]
        new_affinities = np.concatenate((old_affinities, [2.]), axis=0)  # add flag
        assert new_affinities.shape[0] == NUM_NODES + 1
        return new_affinities


def discount_rewards(r, discount_factor):
    """Take 1D float array of rewards and compute discounted reward."""

    gamma = discount_factor
    discounted_r = np.zeros_like(r)
    running_add = 0

    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    return discounted_r.astype(np.float32)


def train(env, policy_net, num_episodes, discount_factor=0.99, saver=None, sess=None):
    """Stochastic PG Algorithm. Optimizes the policy
       function approximator using policy gradient.

    Args:
        env: OpenAI environment.
        policy_net: Policy Function to be optimized
        num_episodes: Number of episodes to run for
        discount_factor: Time-discount factor
        saver: saver object
        sess: Session Object

    Returns:
        An EpisodeStats object with three numpy arrays for obs, acts, rewards.
    """
    train_step = 0
    env.test = sequential
    env.baseline = baseline
    env.raw_reward = raw_reward

    global train_round

    if baseline:
        train_round = "baseline"
        num_episodes = (len(env.train_workloads) * len(env.idx_pairs)) + 1

    if raw_reward and not baseline:
        train_round = 'test'
        num_episodes = 2 * (len(env.train_workloads) * len(env.idx_pairs)) + 1

    # LOG FILE
    log_file = open('training_log_{}.csv'.format(train_round), 'w')
    if not raw_reward:
        log_file.write('episode,target,episode_mean,running_mean,workmix,interf,hm_bad_nodes,prepare_time,episode_type\n')
    else:
        log_file.write('episode,episode_mean,running_mean,workmix,interf,hm_bad_nodes,prepare_time,episode_type\n')

    header = ['episode', 'step']
    header += ['{}_{}'.format(STAT_NAMES[i // NUM_NODES], SAR_NODE_NAMES[i % NUM_NODES]) for i in range(len(SAR_NODE_NAMES) * len(STAT_NAMES))]
    header += ['value_type']
    header += ['new_value_{}'.format(n) for n in SAR_NODE_NAMES]
    header += ['rd_rt', 'rd_tp', 'rd_bw', 'wr_rt', 'wr_tp', 'wr_bw']
    header += ['reward', 'workmix'] + ['prob_' + a for a in ACTIONS] + ['action_taken_for_this_reward', 'bad_nodes', 'interference']

    # STAT FILE
    stat_file = open('stat_log_{}.csv'.format(train_round), 'w')
    stat_file.write(','.join(header) + '\n')

    stats = EpisodeStats(obs=np.zeros((num_episodes * 10, TOTAL_STATS*len(SAR_NODE_NAMES))),
                         acts=np.zeros(num_episodes * 10),
                         cos_values=np.zeros((num_episodes * 10, 6)),
                         rewards=np.zeros(num_episodes * 10))

    # used in computing weighted running avg.
    running_reward = None

    # To collect data to train policy
    states, rewards, targets = [], [], []

    env.batch_size = batch_size

    for episode_number in range(1, num_episodes):
        env.episode_number = episode_number

        observation = env.reset()

        prev_obs = None  # keep track of prev obs
        step = 0
        episode_reward_collect = []

        episode_start = time.time()

        prepare_time = None
        prepare_action_counts = 0  # keep track of how many actions passed before prepare finishes

        for step in itertools.count():

            prepare_action_counts += 1

            train_step += 1
            curr_obs = observation

            state = get_state(curr_obs, prev_obs)  # returns state encoding (5*8, )

            if state is None:
                # observation, reward, done, info = env.step(np.concatenate((np.ones(7), [2.]), axis=0))
                # continue
                prepare_action_counts = 3
                break

            prev_obs = curr_obs

            # get probabdistb. over actions based on observation
            net_input = np.expand_dims(np.asarray(state), axis=0)  # (1, 5*8)

            aprob = policy_net.predict(net_input).ravel()

            action_idx = np.random.choice(np.arange(N_ACTIONS, dtype=np.int32), p=aprob)

            # PREDICTED <-----
            action = ACTIONS[action_idx]

            # get new_values for cluster, last value is flag for weight or affinity
            pre = ""
            post = ""

            if str(env.interference) == "execute_io":
                pre = "io_"
            elif str(env.interference) == "execute_net":
                pre = "net_"

            if str(env.workload) in ["workload1.xml", "workload3.xml", "workload6.xml"]:  # Read, 6 is large
                post = "a"

            elif str(env.workload) in ["workload2.xml", "workload4.xml", "workload5.xml"]:  # Write, 5 is small
                post = "w"

            # FIXED <-----
            fixed_action = pre + post

            if fixed_actions:
                action = fixed_action

            new_values = get_new_values(curr_obs, action_idx, action)

            # Step the environment and get new measurements
            observation, reward, done, info = env.step(new_values)

            # if action == fixed_action:
            #     reward = reward + abs(reward / 1.2)

            print('Step {} reward {}'.format(step, reward))

            if reward != -1:

                prepare_action_counts = 0  # reset count

                if prepare_time is None:
                    prepare_time = time.time() - episode_start

                # For policy update
                states.append(state)
                targets.append(action_idx)
                rewards.append(reward)
                episode_reward_collect.append(reward)

                # For book keeping
                stats.obs[train_step] = curr_obs.ravel()
                stats.acts[train_step] = action_idx
                stats.rewards[train_step] = reward
                stats.cos_values[train_step] = info['COS_VALUES']

                stat_file.write(str(episode_number) + ',')
                stat_file.write(str(train_step) + ',')
                stat_file.write(','.join(curr_obs.ravel().astype(np.str)))
                if new_values[-1] == 1.:
                    stat_file.write(',affinity,')
                elif new_values[-1] == 0.:
                    stat_file.write(',weight,')
                else:
                    stat_file.write(',nop,')
                stat_file.write(','.join(new_values[:-1].astype(np.str)) + ',')
                stat_file.write(','.join(info['COS_VALUES'].astype(np.str)) + ',')
                stat_file.write(str(reward) + ',')
                stat_file.write(str(env.workload) + ',')

                stat_file.write(','.join(aprob.astype(np.str)) + ',') # Action Probs

                stat_file.write(str(action) + ',')
                stat_file.write('|'.join(env.bad_nodes) + ',')
                stat_file.write(str(env.interference) + '\n')
                stat_file.flush()

            if done:
                break

            if prepare_action_counts > 2:
                break

        # Updates:
        if episode_number % batch_size == 0 and not baseline and not raw_reward and not fixed_actions:
            states, rewards, targets = np.vstack(states), np.vstack(rewards), np.vstack(targets)
            # rewards = minmax_scale(rewards, feature_range=(-1, 1))
            discounted_rewards = discount_rewards(rewards, discount_factor)

            # standardize the rewards to be unit normal [helps control the gradient approximator(pg_net) variance]
            discounted_rewards -= np.mean(discounted_rewards)
            discounted_rewards /= np.std(discounted_rewards)

            # Modulate Losees to mod gradient with advantage (PG magic happens right here.)
            print("\nUpdating..Policy..!!")

            loss_after_update = policy_net.magic_update(states,  # to forward and get prediction
                                                        targets,  # to calc losses
                                                        discounted_rewards  # to mod losses
                                                        )
            if saver:  # save after every update
                saver.save(sess, save_ckpt)
                print("SAVED..")

            # Reset after updates
            states, rewards, targets = [], [], []

        # DISPLAY PROGRESS
        episode_reward_total = np.nanmean(episode_reward_collect) * 1000
        episode_type = "good" if prepare_action_counts <= 2 else "bad"

        if episode_type == "good": # update running reward
            running_reward = episode_reward_total if running_reward is None else running_reward * 0.99 + episode_reward_total * 0.01

        print("\nResetting env. Episode Reward Total: {} || Running Mean: {}".format(episode_reward_total,
                                                                                     running_reward))

        if not raw_reward:
            log_file.write('{},{:>3.4f},{:>3.4f},{:>3.4f},{},{},{},{},{}\n'.format(episode_number, env.reward_target,
                                                                         episode_reward_total, running_reward,
                                                                         env.workload, env.interference, env.hm_bad_nodes, prepare_time, episode_type))
        else:
            log_file.write('{},{:>3.4f},{:>3.4f},{},{},{},{},{}\n'.format(episode_number,
                                                                         episode_reward_total, running_reward,
                                                                         env.workload, env.interference, env.hm_bad_nodes, prepare_time, episode_type))

        log_file.flush()

    log_file.close()
    stat_file.close()

    return stats


if __name__ == "__main__":
    tf.reset_default_graph()

    env = gym.make(env_name)
    print('Env {}, Obs Space {}, Action Space {}'.format(env_name, env.observation_space, env.action_space))

    global_step = tf.Variable(0, name="global_step", trainable=False)
    policy = PolicyNet(OBS_SHAPE, N_ACTIONS, fc1_units=fc1_units,  lr=LEARNING_RATE)

    tvars = tf.trainable_variables()
    saver = tf.train.Saver(var_list=tvars)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        if resume:
            saver.restore(sess, restore_ckpt)
            print("RESUMED..")

        stats = train(env=env,
                      policy_net=policy,
                      num_episodes=n_episodes + 1,
                      discount_factor=gamma,
                      saver=saver,
                      sess=sess)
