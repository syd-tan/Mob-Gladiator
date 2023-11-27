from datetime import datetime

class agent_state:
    def __init__(
        self,
        enemy,
        enemy_health,
        agent_health,
        vertical_motion,
        distance_from_enemy,
        attack_cooldown_remaining,
        in_range,
    ):
        self.enemy_mapping = {"Zombie": 0, "Skeleton": 1, "Spider": 2, "Creeper": 3}
        # States for DQN
        self.enemy = self.enemy_mapping[enemy]
        self.enemy_health = enemy_health
        self.agent_health = agent_health
        self.player_vertical_motion = vertical_motion  # 0 - stationary, 1 - upward velocity, -1 - downward velocity
        self.distance_from_enemy = distance_from_enemy
        self.attack_cooldown_remaining = attack_cooldown_remaining
        self.is_in_range = int(in_range)

        # Other values to keep track of
        self.cooldown_completion_time = datetime.now()
        self.damage_dealt = 0
        self.damage_taken = 0
        self.action_performed = None

    def get_state(self):
        return [
            self.enemy,
            self.enemy_health,
            self.agent_health,
            self.player_vertical_motion,
            self.player_vertical_motion,
            self.distance_from_enemy,
            self.attack_cooldown_remaining,
            self.is_in_range,
        ]

    def get_enemy_health(self):
        return self.enemy_health

    def get_agent_health(self):
        return self.agent_health

    def get_attack_cooldown_remaining(self):
        return self.attack_cooldown_remaining

    def get_cooldown_completion_time(self):
        return self.cooldown_completion_time

    def get_damage_dealt(self):
        return self.damage_dealt

    def get_damage_taken(self):
        return self.damage_taken

    def get_action_performed(self):
        return self.action_performed

    def set_cooldown_completion_time(self, cooldown_completion_time):
        self.cooldown_completion_time = cooldown_completion_time

    def set_damage_dealt(self, damage_dealt):
        self.damage_dealt = damage_dealt

    def set_damage_taken(self, damage_taken):
        self.damage_taken = damage_taken

    def set_action_performed(self, action):
        self.action_performed = action