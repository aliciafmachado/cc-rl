from gym.envs.registration import register

register(
    id='cl-v0',
    entry_point='gym_cl.envs:ClEnv',
)