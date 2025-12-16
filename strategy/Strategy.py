import math
import numpy as np
from math_ops.Math_Ops import Math_Ops as M


class Strategy():
    def __init__(self, world):
        self.play_mode = world.play_mode
        self.robot_model = world.robot  
        self.my_head_pos_2d = self.robot_model.loc_head_position[:2]
        self.player_unum = self.robot_model.unum
        self.mypos = (world.teammates[self.player_unum-1].state_abs_pos[0],
                      world.teammates[self.player_unum-1].state_abs_pos[1])
       
        self.side = 1
        if world.team_side_is_left:
            self.side = 0

        self.teammate_positions = [teammate.state_abs_pos[:2] if teammate.state_abs_pos is not None 
                                    else None
                                    for teammate in world.teammates]
        
        self.opponent_positions = [opponent.state_abs_pos[:2] if opponent.state_abs_pos is not None 
                                    else None
                                    for opponent in world.opponents]

        self.team_dist_to_ball = None
        self.team_dist_to_oppGoal = None
        self.opp_dist_to_ball = None

        self.prev_important_positions_and_values = None
        self.curr_important_positions_and_values = None
        self.point_preferences = None
        self.combined_threat_and_definedPositions = None

        self.my_ori = self.robot_model.imu_torso_orientation
        self.ball_2d = world.ball_abs_pos[:2]
        self.ball_vec = self.ball_2d - self.my_head_pos_2d
        self.ball_dir = M.vector_angle(self.ball_vec)
        self.ball_dist = np.linalg.norm(self.ball_vec)
        self.ball_sq_dist = self.ball_dist * self.ball_dist
        self.ball_speed = np.linalg.norm(world.get_ball_abs_vel(6)[:2])
        
        # Enhanced ball velocity tracking
        self.ball_velocity_estimate = world.get_ball_abs_vel(6)[:2]
        
        self.goal_dir = M.target_abs_angle(self.ball_2d,(15.05,0))
        self.PM_GROUP = world.play_mode_group
        self.slow_ball_pos = world.get_predicted_ball_pos(0.5)

        # Distance calculations with more lenient time window
        self.teammates_ball_sq_dist = [np.sum((p.state_abs_pos[:2] - self.slow_ball_pos) ** 2)
                                  if p.state_last_update != 0 and (world.time_local_ms - p.state_last_update <= 400 or p.is_self) and not p.state_fallen
                                  else 1000
                                  for p in world.teammates]

        self.opponents_ball_sq_dist = [np.sum((p.state_abs_pos[:2] - self.slow_ball_pos) ** 2)
                                  if p.state_last_update != 0 and world.time_local_ms - p.state_last_update <= 400 and not p.state_fallen
                                  else 1000
                                  for p in world.opponents]

        self.min_teammate_ball_sq_dist = min(self.teammates_ball_sq_dist)
        self.min_teammate_ball_dist = math.sqrt(self.min_teammate_ball_sq_dist)
        self.min_opponent_ball_dist = math.sqrt(min(self.opponents_ball_sq_dist))

        self.active_player_unum = self.teammates_ball_sq_dist.index(self.min_teammate_ball_sq_dist) + 1

        self.my_desired_position = self.mypos
        self.my_desired_orientation = self.ball_dir


    def IsFormationReady(self, point_preferences):
        """Check if team is in formation - more lenient"""
        is_formation_ready = True
        for i in range(1, 6):
            if i != self.active_player_unum: 
                teammate_pos = self.teammate_positions[i-1]
                if teammate_pos is not None:
                    distance = np.sum((teammate_pos - point_preferences[i]) **2)
                    if distance > 0.5:  # More lenient threshold
                        is_formation_ready = False
        return is_formation_ready


    def GetDirectionRelativeToMyPositionAndTarget(self, target):
        """Get angle to target"""
        target_vec = target - self.my_head_pos_2d
        target_dir = M.vector_angle(target_vec)
        return target_dir
    
    
    def GetClosestOpponentToPosition(self, position):
        """Find closest opponent to a position"""
        min_dist = float('inf')
        closest_opp = None
        
        for opp_pos in self.opponent_positions:
            if opp_pos is None:
                continue
            dist = np.linalg.norm(np.array(opp_pos) - np.array(position))
            if dist < min_dist:
                min_dist = dist
                closest_opp = opp_pos
        
        return closest_opp, min_dist