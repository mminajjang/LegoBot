from gym.envs.registration import register

register(
    id='KukaLegoSorter-v0', 
    entry_point='kuka_lego_sorter.envs:KukaLegoEnv'
)
