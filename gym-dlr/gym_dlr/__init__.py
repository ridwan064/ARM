import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Dlr-v0',
    entry_point='gym_dlr.envs:DlrEnv',
    # timestep_limit=1000,
    # reward_threshold=1.0,
    # nondeterministic=True,
)

# register(
#     id='',
#     entry_point='',
#     timestep_limit=1000,
#     reward_threshold=10.0,
#     nondeterministic=True,
# )
#
# register(
#     id='',
#     entry_point='',
#     timestep_limit=1000,
#     reward_threshold=8.0,
#     nondeterministic=True,
# )
