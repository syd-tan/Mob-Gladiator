class agent_state:
    def __init__(
        self,
        enemy,
        enemy_health,
        agent_health,
        vertical_motion,
        distance_from_enemy,
        attack_cooldown, 
        in_range,
    ):
        self.enemy = enemy
        self.enemy_health = enemy_health
        self.agent_health = agent_health

        # 0 - stationary, 1 - upward velocity, -1 - downward velocity
        self.player_vertical_motion = vertical_motion
        self.distance_from_enemy = distance_from_enemy
        self.attack_cooldown = attack_cooldown
        self.is_in_range = in_range

    def get_enemy_health(self):
        return self.enemy_health

    def get_agent_health(self):
        return self.agent_health

    def get_attack_cooldown(self):
        return self.attack_cooldown
