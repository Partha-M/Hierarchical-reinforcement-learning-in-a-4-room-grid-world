from gym.envs.registration import register

register(
    id='RoomGridWorld-v0',
    entry_point='RoomGridworld.envs:FourRoomEnv',
)