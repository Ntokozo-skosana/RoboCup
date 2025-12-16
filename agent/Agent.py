from agent.Base_Agent import Base_Agent
from math_ops.Math_Ops import Math_Ops as M
import math
import numpy as np

from strategy.Assignment import role_assignment 
from strategy.Strategy import Strategy 
from formation.Formation import GenerateDynamicFormation


class Agent(Base_Agent):
    def __init__(self, host:str, agent_port:int, monitor_port:int, unum:int,
                 team_name:str, enable_log, enable_draw, wait_for_server=True, is_fat_proxy=False) -> None:
        
        robot_type = (0,1,1,1,2)[unum-1]
        super().__init__(host, agent_port, monitor_port, unum, robot_type, team_name, enable_log, enable_draw, True, wait_for_server, None)

        self.enable_draw = enable_draw
        self.state = 0
        self.kick_direction = 0
        self.kick_distance = 0
        self.fat_proxy_cmd = "" if is_fat_proxy else None
        self.fat_proxy_walk = np.zeros(3)
        self.init_pos = ([-13,0], [-8,-3], [-8,3], [-3,0], [2,0])[unum-1]
        
        # Enhanced tracking
        self.last_ball_pos = None
        self.ball_velocity_estimate = np.zeros(2)
        self.frames_tracking = 0


    def beam(self, avoid_center_circle=False):
        r = self.world.robot
        pos = self.init_pos[:]
        self.state = 0

        if avoid_center_circle and np.linalg.norm(self.init_pos) < 2.5:
            pos[0] = -2.3 

        if np.linalg.norm(pos - r.loc_head_position[:2]) > 0.1 or self.behavior.is_ready("Get_Up"):
            self.scom.commit_beam(pos, M.vector_angle((-pos[0],-pos[1])))
        else:
            if self.fat_proxy_cmd is None:
                self.behavior.execute("Zero_Bent_Knees_Auto_Head")
            else:
                self.fat_proxy_cmd += "(proxy dash 0 0 0)"
                self.fat_proxy_walk = np.zeros(3)


    def estimate_ball_velocity(self, strategyData):
        """Track ball velocity for interception"""
        current_ball = strategyData.ball_2d
        
        if self.last_ball_pos is not None:
            velocity = current_ball - self.last_ball_pos
            # Smooth velocity estimate
            self.ball_velocity_estimate = 0.7 * self.ball_velocity_estimate + 0.3 * velocity
            self.frames_tracking += 1
        
        self.last_ball_pos = current_ball.copy()
        return self.ball_velocity_estimate


    def predict_ball_intercept_point(self, strategyData, frames_ahead=8):
        """Predict where ball will be for interception"""
        ball_vel = self.estimate_ball_velocity(strategyData)
        ball_speed = np.linalg.norm(ball_vel)
        
        # Only predict if ball is moving
        if ball_speed > 0.05 and self.frames_tracking > 3:
            # Apply friction decay
            predicted_pos = strategyData.ball_2d.copy()
            current_vel = ball_vel.copy()
            
            for _ in range(frames_ahead):
                predicted_pos += current_vel
                current_vel *= 0.92  # friction
            
            # Clamp to field
            predicted_pos[0] = np.clip(predicted_pos[0], -14.5, 14.5)
            predicted_pos[1] = np.clip(predicted_pos[1], -9.5, 9.5)
            
            return predicted_pos
        
        return strategyData.ball_2d


    def move(self, target_2d=(0,0), orientation=None, is_orientation_absolute=True,
             avoid_obstacles=True, priority_unums=[], is_aggressive=False, timeout=3000):
        r = self.world.robot

        if self.fat_proxy_cmd is not None:
            self.fat_proxy_move(target_2d, orientation, is_orientation_absolute)
            return

        # ELITE: Be aggressive near ball
        ball_2d = self.world.ball_abs_pos[:2]
        dist_to_ball = np.linalg.norm(target_2d - ball_2d)
        
        if dist_to_ball < 2:
            is_aggressive = True
            timeout = 1000

        if avoid_obstacles:
            target_2d, _, distance_to_final_target = self.path_manager.get_path_to_target(
                target_2d, priority_unums=priority_unums, is_aggressive=is_aggressive, timeout=timeout)
        else:
            distance_to_final_target = np.linalg.norm(target_2d - r.loc_head_position[:2])

        self.behavior.execute("Walk", target_2d, True, orientation, is_orientation_absolute, distance_to_final_target)


    def kickTarget(self, strategyData, mypos_2d=(0,0), target_2d=(0,0), abort=False, enable_pass_command=False):
        """ELITE: Optimized for speed"""
        vector_to_target = np.array(target_2d) - np.array(mypos_2d)
        kick_distance = np.linalg.norm(vector_to_target)
        direction_radians = np.arctan2(vector_to_target[1], vector_to_target[0])
        kick_direction = np.degrees(direction_radians)

        if strategyData.min_opponent_ball_dist < 1.45 and enable_pass_command:
            self.scom.commit_pass_command()

        self.kick_direction = kick_direction
        self.kick_distance = kick_distance

        if self.fat_proxy_cmd is None:
            return self.behavior.execute("Basic_Kick", self.kick_direction, abort)
        else:
            return self.fat_proxy_kick()


    def think_and_send(self):
        behavior = self.behavior
        strategyData = Strategy(self.world)
        d = self.world.draw

        if strategyData.play_mode == self.world.M_GAME_OVER:
            pass
        elif strategyData.PM_GROUP == self.world.MG_ACTIVE_BEAM:
            self.beam()
        elif strategyData.PM_GROUP == self.world.MG_PASSIVE_BEAM:
            self.beam(True)
        elif self.state == 1 or (behavior.is_ready("Get_Up") and self.fat_proxy_cmd is None):
            self.state = 0 if behavior.execute("Get_Up") else 1
        else:
            if strategyData.play_mode != self.world.M_BEFORE_KICKOFF:
                self.select_skill(strategyData)
            else:
                pass

        self.radio.broadcast()

        if self.fat_proxy_cmd is None:
            self.scom.commit_and_send( strategyData.robot_model.get_command() )
        else:
            self.scom.commit_and_send( self.fat_proxy_cmd.encode() ) 
            self.fat_proxy_cmd = ""


    def handle_set_piece(self, strategyData):
        """ELITE: Aggressive set pieces"""
        W = self.world
        pm = strategyData.play_mode
        
        if pm == W.M_OUR_KICKOFF:
            if strategyData.active_player_unum == strategyData.player_unum:
                # Quick forward kick
                forward_players = []
                for i in range(1, 5):
                    if i == strategyData.player_unum:
                        continue
                    tm_pos = strategyData.teammate_positions[i]
                    if tm_pos is not None and tm_pos[0] > 3:
                        forward_players.append(tm_pos)
                
                if forward_players:
                    target = max(forward_players, key=lambda p: p[0])
                else:
                    target = np.array([8, 0])
                return self.kickTarget(strategyData, strategyData.mypos, target)
            else:
                return self.move(strategyData.my_desired_position, orientation=strategyData.ball_dir)
        
        elif pm == W.M_THEIR_KICKOFF:
            # Press immediately
            if strategyData.player_unum != 1:
                press_pos = np.array([min(-1, strategyData.my_desired_position[0]), 
                                     strategyData.my_desired_position[1]])
                return self.move(press_pos, orientation=strategyData.ball_dir)
            else:
                return self.move(np.array([-13.5, 0]), orientation=strategyData.ball_dir)
        
        elif pm == W.M_OUR_KICK_IN:
            if strategyData.active_player_unum == strategyData.player_unum:
                target = self.find_best_pass_target(strategyData)
                return self.kickTarget(strategyData, strategyData.mypos, target)
            else:
                return self.move(strategyData.my_desired_position, orientation=strategyData.ball_dir)
        
        elif pm == W.M_THEIR_KICK_IN:
            # Press the ball
            if strategyData.player_unum > 1 and strategyData.ball_dist < 4:
                return self.move(strategyData.ball_2d, orientation=strategyData.ball_dir, is_aggressive=True)
            return self.move(strategyData.my_desired_position, orientation=strategyData.ball_dir)
        
        elif pm == W.M_OUR_CORNER_KICK:
            if strategyData.active_player_unum == strategyData.player_unum:
                return self.kickTarget(strategyData, strategyData.mypos, (15, 0))
            else:
                if strategyData.player_unum == 1:
                    return self.move(np.array([-13, 0]), orientation=strategyData.ball_dir)
                else:
                    # Swarm goal
                    attack_pos = np.array([12 + np.random.rand() * 2, 
                                          (strategyData.player_unum - 3) * 2.2])
                    return self.move(attack_pos, orientation=strategyData.ball_dir)
        
        elif pm == W.M_OUR_GOAL_KICK:
            if strategyData.active_player_unum == strategyData.player_unum:
                target = self.find_best_pass_target(strategyData)
                return self.kickTarget(strategyData, strategyData.mypos, target)
            else:
                return self.move(strategyData.my_desired_position, orientation=strategyData.ball_dir)
        
        elif pm == W.M_THEIR_CORNER_KICK:
            if strategyData.player_unum == 1:
                return self.move(np.array([-14, 0]), orientation=strategyData.ball_dir)
            else:
                # Defend goal line
                defend_pos = np.array([-12, (strategyData.player_unum - 3) * 2.5])
                return self.move(defend_pos, orientation=strategyData.ball_dir)
        
        return self.move(strategyData.my_desired_position, orientation=strategyData.ball_dir)


    def find_best_pass_target(self, strategyData):
        """ELITE: Advanced passing with space awareness"""
        my_pos = np.array(strategyData.mypos)
        goal_pos = np.array([15, 0])
        best_target = goal_pos
        best_score = -10000
        
        for i in range(5):
            if i + 1 == strategyData.player_unum or i == 0:
                continue
            
            tm_pos = strategyData.teammate_positions[i]
            if tm_pos is None:
                continue
            
            tm_pos = np.array(tm_pos)
            
            # Multi-factor scoring
            forward_score = tm_pos[0] * 4  # Heavily favor forward
            dist_to_goal = np.linalg.norm(tm_pos - goal_pos)
            goal_score = -dist_to_goal * 0.6
            pass_dist = np.linalg.norm(tm_pos - my_pos)
            
            # Distance sweet spot
            if pass_dist < 0.8:
                dist_score = -15
            elif 3 < pass_dist < 10:
                dist_score = 5
            elif pass_dist < 14:
                dist_score = 2
            else:
                dist_score = -pass_dist * 0.5
            
            # Clear path bonus
            path_clear = self.is_path_clear(my_pos, tm_pos, strategyData)
            clear_score = 10 if path_clear else -6
            
            # Ahead bonus
            ahead_score = 7 if tm_pos[0] > my_pos[0] + 1.5 else -3
            
            # Open space bonus
            open_score = 0
            if strategyData.opponent_positions:
                min_opp_dist = float('inf')
                for opp_pos in strategyData.opponent_positions:
                    if opp_pos is None:
                        continue
                    dist = np.linalg.norm(np.array(opp_pos) - tm_pos)
                    min_opp_dist = min(min_opp_dist, dist)
                
                if min_opp_dist > 3.5:
                    open_score = 6
                elif min_opp_dist > 2.5:
                    open_score = 3
            
            # Wing bonus (spread the field)
            wing_score = 0
            if abs(tm_pos[1]) > 4:
                wing_score = 2
            
            score = forward_score + goal_score + dist_score + clear_score + ahead_score + open_score + wing_score
            
            if score > best_score:
                best_score = score
                best_target = tm_pos
        
        # Shoot if no good pass
        if best_score < 2:
            dist_to_goal = np.linalg.norm(my_pos - goal_pos)
            if dist_to_goal < 20 and my_pos[0] > -2:
                return goal_pos
        
        return best_target


    def is_path_clear(self, start, end, strategyData, threshold=0.85):
        """Check path clearance"""
        if not strategyData.opponent_positions:
            return True
        
        start = np.array(start)
        end = np.array(end)
        path_vec = end - start
        path_len = np.linalg.norm(path_vec)
        
        if path_len < 0.1:
            return True
        
        for opp_pos in strategyData.opponent_positions:
            if opp_pos is None:
                continue
            
            opp_pos = np.array(opp_pos)
            t = max(0, min(1, np.dot(opp_pos - start, path_vec) / (path_len * path_len + 0.001)))
            closest_point = start + t * path_vec
            dist = np.linalg.norm(opp_pos - closest_point)
            
            if dist < threshold:
                return False
        
        return True


    def should_dribble(self, strategyData):
        """Enhanced dribble decision"""
        my_pos = np.array(strategyData.mypos)
        ball_pos = np.array(strategyData.ball_2d)
        goal_pos = np.array([15, 0])
        
        # Always dribble if very close to goal
        if np.linalg.norm(ball_pos - goal_pos) < 5:
            return True
        
        has_good_pass = False
        
        for i in range(5):
            if i + 1 == strategyData.player_unum or i == 0:
                continue
            
            tm_pos = strategyData.teammate_positions[i]
            if tm_pos is None:
                continue
            
            tm_pos = np.array(tm_pos)
            pass_dist = np.linalg.norm(tm_pos - my_pos)
            
            if (3 < pass_dist < 13 and 
                tm_pos[0] > ball_pos[0] + 0.5 and
                self.is_path_clear(ball_pos, tm_pos, strategyData)):
                has_good_pass = True
                break
        
        return not has_good_pass


    def dribble_to_goal(self, strategyData):
        """ELITE: Intelligent dribbling with obstacle avoidance"""
        ball_pos = np.array(strategyData.ball_2d)
        goal_pos = np.array([15, 0])
        my_pos = np.array(strategyData.mypos)
        
        direction_to_goal = goal_pos - ball_pos
        direction_to_goal = direction_to_goal / (np.linalg.norm(direction_to_goal) + 0.001)
        
        # Check for opponents in path
        best_direction = direction_to_goal.copy()
        min_threat = float('inf')
        
        if strategyData.opponent_positions:
            for angle_offset in [0, -20, 20, -35, 35]:
                angle_rad = np.arctan2(direction_to_goal[1], direction_to_goal[0]) + np.radians(angle_offset)
                test_dir = np.array([np.cos(angle_rad), np.sin(angle_rad)])
                test_target = ball_pos + test_dir * 1.5
                
                # Find closest opponent to this path
                threat_level = 0
                for opp_pos in strategyData.opponent_positions:
                    if opp_pos is None:
                        continue
                    opp_dist = np.linalg.norm(np.array(opp_pos) - test_target)
                    if opp_dist < 1.5:
                        threat_level += (1.5 - opp_dist) * 2
                
                if threat_level < min_threat:
                    min_threat = threat_level
                    best_direction = test_dir
        
        # Dribble target
        dribble_dist = 0.55 if min_threat > 0.5 else 0.4
        dribble_target = ball_pos + best_direction * dribble_dist
        dribble_target[0] = np.clip(dribble_target[0], -14, 14)
        dribble_target[1] = np.clip(dribble_target[1], -9, 9)
        
        orientation = np.degrees(np.arctan2(best_direction[1], best_direction[0]))
        
        return self.move(dribble_target, orientation=orientation, 
                        is_orientation_absolute=True, avoid_obstacles=True, is_aggressive=True)


    def is_power_shot_range(self, strategyData):
        """Power shot range"""
        ball_pos = np.array(strategyData.ball_2d)
        goal_pos = np.array([15, 0])
        dist = np.linalg.norm(ball_pos - goal_pos)
        return dist < 7 and ball_pos[0] > 7.5


    def should_shoot(self, strategyData):
        """ELITE: Very aggressive shooting"""
        ball_pos = np.array(strategyData.ball_2d)
        goal_pos = np.array([15, 0])
        dist_to_goal = np.linalg.norm(ball_pos - goal_pos)
        
        # Extended shooting range
        if dist_to_goal > 20 or ball_pos[0] < -4 or abs(ball_pos[1]) > 11:
            return False
        
        # Shoot unless teammate is MUCH better positioned
        for i, tm_pos in enumerate(strategyData.teammate_positions):
            if i + 1 == strategyData.player_unum or tm_pos is None or i == 0:
                continue
            
            tm_pos = np.array(tm_pos)
            tm_dist = np.linalg.norm(tm_pos - goal_pos)
            
            # Need to be 7m closer with clear path
            if (tm_dist < dist_to_goal - 7 and 
                tm_pos[0] > ball_pos[0] + 4 and 
                self.is_path_clear(ball_pos, tm_pos, strategyData)):
                return False
        
        return True


    def should_challenge_ball(self, strategyData):
        """Determine if support player should challenge for ball"""
        my_dist = strategyData.ball_dist
        active_dist = strategyData.min_teammate_ball_dist
        
        # Challenge if much closer
        if my_dist < active_dist - 0.2 and my_dist < 1.2:
            return True
        
        # Intercept moving ball
        ball_vel = self.estimate_ball_velocity(strategyData)
        ball_speed = np.linalg.norm(ball_vel)
        
        if ball_speed > 0.15:
            intercept_pos = self.predict_ball_intercept_point(strategyData, frames_ahead=6)
            intercept_dist = np.linalg.norm(intercept_pos - np.array(strategyData.mypos))
            
            if intercept_dist < 0.8:
                return True
        
        return False


    def press_opponent(self, strategyData):
        """Press nearest opponent with ball"""
        ball_pos = np.array(strategyData.ball_2d)
        
        # Find nearest opponent to ball
        nearest_opp = None
        nearest_dist = float('inf')
        
        for opp_pos in strategyData.opponent_positions:
            if opp_pos is None:
                continue
            opp_dist = np.linalg.norm(np.array(opp_pos) - ball_pos)
            if opp_dist < nearest_dist:
                nearest_dist = opp_dist
                nearest_opp = opp_pos
        
        if nearest_opp is not None and nearest_dist < 0.7:
            # Press the opponent
            press_target = np.array(nearest_opp)
            return self.move(press_target, orientation=strategyData.ball_dir, is_aggressive=True)
        
        # Press ball
        return self.move(ball_pos, orientation=strategyData.ball_dir, is_aggressive=True)


    def select_skill(self, strategyData):
        drawer = self.world.draw
        W = self.world
        
        if strategyData.PM_GROUP in [W.MG_OUR_KICK, W.MG_THEIR_KICK]:
            return self.handle_set_piece(strategyData)
        
        formation_positions = GenerateDynamicFormation(strategyData)
        point_preferences = role_assignment(strategyData.teammate_positions, formation_positions)
        strategyData.my_desired_position = point_preferences[strategyData.player_unum]
        
        drawer.line(strategyData.mypos, strategyData.my_desired_position, 2, drawer.Color.blue, "target_line")
        
        am_active = strategyData.active_player_unum == strategyData.player_unum
        ball_pos = np.array(strategyData.ball_2d)
        my_pos = np.array(strategyData.mypos)
        
        # GOALKEEPER - Enhanced
        if strategyData.player_unum == 1:
            if am_active and ball_pos[0] < -8 and strategyData.ball_dist < 5:
                if self.is_power_shot_range(strategyData):
                    drawer.annotation((0, 10.5), "GK: POWER!", drawer.Color.red, "status")
                    return self.kickTarget(strategyData, strategyData.mypos, (15, 0))
                elif self.should_shoot(strategyData):
                    drawer.annotation((0, 10.5), "GK: Shoot", drawer.Color.orange, "status")
                    return self.kickTarget(strategyData, strategyData.mypos, (15, 0))
                elif self.should_dribble(strategyData):
                    drawer.annotation((0, 10.5), "GK: Dribble", drawer.Color.cyan, "status")
                    return self.dribble_to_goal(strategyData)
                else:
                    drawer.annotation((0, 10.5), "GK: Pass", drawer.Color.green, "status")
                    target = self.find_best_pass_target(strategyData)
                    return self.kickTarget(strategyData, strategyData.mypos, target)
            else:
                # Dynamic GK positioning
                gk_x = max(-14, min(-13.5, ball_pos[0] * 0.15 - 13.5))
                gk_y = np.clip(ball_pos[1] * 0.35, -2.5, 2.5)
                gk_pos = np.array([gk_x, gk_y])
                return self.move(gk_pos, orientation=strategyData.ball_dir)
        
        # ACTIVE PLAYER - Enhanced Decision Tree
        if am_active:
            # 1. Power shot
            if self.is_power_shot_range(strategyData):
                drawer.annotation((0, 10.5), "⚡ POWER!", drawer.Color.red, "status")
                return self.kickTarget(strategyData, strategyData.mypos, (15, 0))
            
            # 2. Regular shot
            elif self.should_shoot(strategyData):
                drawer.annotation((0, 10.5), "🎯 SHOOT!", drawer.Color.red, "status")
                drawer.annotation((0, 9.5), f"{np.linalg.norm(ball_pos - np.array([15,0])):.1f}m", 
                                 drawer.Color.white, "dist")
                return self.kickTarget(strategyData, strategyData.mypos, (15, 0))
            
            # 3. Dribble
            elif self.should_dribble(strategyData):
                drawer.annotation((0, 10.5), "🏃 DRIBBLE", drawer.Color.cyan, "status")
                return self.dribble_to_goal(strategyData)
            
            # 4. Pass
            else:
                drawer.annotation((0, 10.5), "⚽ PASS", drawer.Color.green, "status")
                target = self.find_best_pass_target(strategyData)
                drawer.line(strategyData.mypos, target, 2, drawer.Color.green, "pass_line")
                return self.kickTarget(strategyData, strategyData.mypos, target, enable_pass_command=True)
        
        # SUPPORT PLAYERS - Enhanced with interception and pressing
        else:
            drawer.clear("status")
            drawer.clear("pass_line")
            drawer.clear("dist")
            
            # 1. Challenge/Intercept ball
            if self.should_challenge_ball(strategyData):
                intercept_pos = self.predict_ball_intercept_point(strategyData, frames_ahead=6)
                drawer.annotation((my_pos[0], my_pos[1] + 0.5), "CHALLENGE!", drawer.Color.yellow, f"p{strategyData.player_unum}")
                
                # Use shooting logic if we get the ball
                if strategyData.ball_dist < 0.5:
                    if self.is_power_shot_range(strategyData):
                        return self.kickTarget(strategyData, strategyData.mypos, (15, 0))
                    elif self.should_shoot(strategyData):
                        return self.kickTarget(strategyData, strategyData.mypos, (15, 0))
                    else:
                        target = self.find_best_pass_target(strategyData)
                        return self.kickTarget(strategyData, strategyData.mypos, target)
                
                return self.move(intercept_pos, orientation=strategyData.ball_dir, is_aggressive=True)
            
            # 2. Press opponent when they have ball
            if ball_pos[0] > -8 and strategyData.ball_dist < 4 and strategyData.min_opponent_ball_dist < 1.5:
                drawer.annotation((my_pos[0], my_pos[1] + 0.5), "PRESS", drawer.Color.orange, f"p{strategyData.player_unum}")
                return self.press_opponent(strategyData)
            
            # 3. Defend if ball in danger zone
            if ball_pos[0] < -6 and np.linalg.norm(ball_pos - my_pos) < 6:
                goal_pos = np.array([-15, 0])
                ball_to_goal = goal_pos - ball_pos
                ball_to_goal_norm = ball_to_goal / (np.linalg.norm(ball_to_goal) + 0.001)
                defend_pos = ball_pos + ball_to_goal_norm * 2.2
                defend_pos[0] = np.clip(defend_pos[0], -13.5, 10)
                defend_pos[1] = np.clip(defend_pos[1], -9, 9)
                drawer.annotation((my_pos[0], my_pos[1] + 0.5), "DEFEND", drawer.Color.blue, f"p{strategyData.player_unum}")
                return self.move(defend_pos, orientation=strategyData.ball_dir)
            
            # 4. Go to attacking position (closer than formation suggests)
            if ball_pos[0] > 3:
                # Push up more aggressively
                attack_pos = strategyData.my_desired_position.copy()
                attack_pos[0] += 2
                attack_pos[0] = np.clip(attack_pos[0], -13, 13)
                return self.move(attack_pos, orientation=strategyData.ball_dir)
            
            # 5. Default to formation position
            return self.move(strategyData.my_desired_position, orientation=strategyData.ball_dir)


    # Fat proxy methods
    def fat_proxy_kick(self):
        w = self.world
        r = self.world.robot 
        ball_2d = w.ball_abs_pos[:2]
        my_head_pos_2d = r.loc_head_position[:2]

        if np.linalg.norm(ball_2d - my_head_pos_2d) < 0.25:
            self.fat_proxy_cmd += f"(proxy kick 10 {M.normalize_deg( self.kick_direction  - r.imu_torso_orientation ):.2f} 20)" 
            self.fat_proxy_walk = np.zeros(3)
            return True
        else:
            self.fat_proxy_move(ball_2d-(-0.1,0), None, True)
            return False

    def fat_proxy_move(self, target_2d, orientation, is_orientation_absolute):
        r = self.world.robot
        target_dist = np.linalg.norm(target_2d - r.loc_head_position[:2])
        target_dir = M.target_rel_angle(r.loc_head_position[:2], r.imu_torso_orientation, target_2d)

        if target_dist > 0.1 and abs(target_dir) < 8:
            self.fat_proxy_cmd += (f"(proxy dash {100} {0} {0})")
            return

        if target_dist < 0.1:
            if is_orientation_absolute:
                orientation = M.normalize_deg( orientation - r.imu_torso_orientation )
            target_dir = np.clip(orientation, -60, 60)
            self.fat_proxy_cmd += (f"(proxy dash {0} {0} {target_dir:.1f})")
        else:
            self.fat_proxy_cmd += (f"(proxy dash {20} {0} {target_dir:.1f})")