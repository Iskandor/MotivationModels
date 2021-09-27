from gym.envs.registration import register

register(
    id='bitflip-8-v0',
    entry_point='gym_bitflip.envs:BitFlipEnv8',
)

register(
    id='bitflip-12-v0',
    entry_point='gym_bitflip.envs:BitFlipEnv12',
)

register(
    id='bitflip-16-v0',
    entry_point='gym_bitflip.envs:BitFlipEnv16',
)
