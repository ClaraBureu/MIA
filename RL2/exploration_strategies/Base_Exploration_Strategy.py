class Base_Exploration_Strategy(object):
    """Base exploration strategy that all exploration strategies inherit from"""
    def __init__(self, config):
        self.config = config
        
    def perturb_action_for_exploration_purposes(self, action_info):
        """Perturbs the action for exploration purposes"""
        raise NotImplementedError("Must be implemented by the child class")
        
    def add_exploration_rewards(self, reward_info):
        """Adds rewards to encourage exploration"""
        return reward_info["reward"]
    
    def reset(self):
        """Resets the strategy"""
        raise NotImplementedError("Must be implemented by the child class")
