import numpy as np

def GenerateBasicFormation():
    """Basic balanced formation"""
    formation = [
        np.array([-13, 0]),
        np.array([-7, -2]),
        np.array([-7, 2]),
        np.array([0, 0]),
        np.array([7, 0])
    ]
    return formation


def GenerateDynamicFormation(strategyData):
    """
    ELITE: Ultra-aggressive adaptive formations
    Key improvements:
    - Tighter marking near ball
    - More aggressive pushing when attacking
    - Better defensive coverage
    - Dynamic wing play
    """
    ball_x = strategyData.ball_2d[0]
    ball_y = np.clip(strategyData.ball_2d[1], -8, 8)
    
    # Goalkeeper - dynamic positioning
    gk_x = max(-14, min(-13.5, ball_x * 0.12 - 13.5))
    gk_y = np.clip(ball_y * 0.35, -2.5, 2.5)
    gk_pos = np.array([gk_x, gk_y])
    
    # CRITICAL DEFENSE (ball at our goal)
    if ball_x < -11:
        formation = [
            gk_pos,
            np.array([-12, np.clip(ball_y * 0.8, -3, 3)]),      # Emergency defender
            np.array([-9, np.clip(-ball_y * 0.6, -4, 4)]),      # Cover defender  
            np.array([-7, np.clip(ball_y * 0.3, -3, 3)]),       # Sweeper
            np.array([-4, 0])                                    # Counter-attack ready
        ]
    
    # EXTREME DEFENSIVE (ball very close to our goal)
    elif ball_x < -8:
        formation = [
            gk_pos,
            np.array([-10, np.clip(ball_y * 0.75, -3.5, 3.5)]), # Press ball
            np.array([-7, np.clip(-ball_y * 0.5, -4, 4)]),      # Support defender
            np.array([-4, np.clip(ball_y * 0.4, -3, 3)]),       # Defensive mid
            np.array([0, 0])                                     # Forward ready
        ]
    
    # DEFENSIVE (ball in our half)
    elif ball_x < -3:
        formation = [
            gk_pos,
            np.array([-7, np.clip(ball_y * 0.8, -4, 4)]),       # Active defender
            np.array([-5, np.clip(-ball_y * 0.5, -4, 4)]),      # Cover
            np.array([-1, np.clip(ball_y * 0.5, -3.5, 3.5)]),   # Transition mid
            np.array([4, np.clip(ball_y * 0.3, -3, 3)])         # Forward
        ]
    
    # BALANCED (ball near center)
    elif ball_x < 3:
        formation = [
            gk_pos,
            np.array([-5, -3.5]),                               # Left back
            np.array([-5, 3.5]),                                # Right back
            np.array([ball_x + 1.5, np.clip(ball_y * 0.8, -4, 4)]),  # Support ball
            np.array([7, np.clip(ball_y * 0.6, -3.5, 3.5)])    # Striker
        ]
    
    # ATTACKING (ball in opponent half)
    elif ball_x < 8:
        formation = [
            gk_pos,
            np.array([-3, -3.5]),                               # Push up left
            np.array([-3, 3.5]),                                # Push up right
            np.array([5, np.clip(ball_y * 0.85, -4.5, 4.5)]),   # Attacking mid
            np.array([11, np.clip(ball_y * 0.65, -3.5, 3.5)])   # High striker
        ]
    
    # ULTRA ATTACK (ball near opponent goal)
    else:
        # All-out attack formation
        formation = [
            gk_pos,
            np.array([-1, -3.5]),                               # Very high backs
            np.array([-1, 3.5]),
            np.array([9, np.clip(ball_y * 0.9, -5, 5)]),        # Attack support
            np.array([13, np.clip(ball_y * 0.7, -3, 3)])        # Striker at goal
        ]
    
    # Additional adjustments based on ball movement
    ball_speed = np.linalg.norm(strategyData.ball_velocity_estimate) if hasattr(strategyData, 'ball_velocity_estimate') else 0
    
    # If ball moving fast toward our goal, drop back
    if ball_speed > 0.3 and ball_x < 0:
        for i in range(1, 5):  # Don't move GK
            formation[i][0] -= 1.5
            formation[i][0] = max(formation[i][0], -12)
    
    # Ensure no collisions and field bounds
    for i in range(5):
        formation[i][0] = np.clip(formation[i][0], -14, 14)
        formation[i][1] = np.clip(formation[i][1], -9.5, 9.5)
    
    return formation