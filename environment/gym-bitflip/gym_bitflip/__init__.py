from gym.envs.registration import register

register(
    id='bitflip-v0',
    entry_point='gym_bitflip.envs:BitFlipEnv',
)
