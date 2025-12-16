import numpy as np

def role_assignment(teammate_positions, formation_positions): 

    n = len(teammate_positions)
    
    player_prefs = {}
    for i in range(n):
        p_pos = teammate_positions[i]
        dists = []
        
        for j in range(n):
            f_pos = formation_positions[j]
            d = np.sqrt((f_pos[0] - p_pos[0])**2 + (f_pos[1] - p_pos[1])**2)
            dists.append((j, d))
        
        dists.sort(key=lambda x: x[1])
        player_prefs[i] = [x[0] for x in dists]
    
    role_prefs = {}
    for j in range(n):
        f_pos = formation_positions[j]
        dists = []
        
        for i in range(n):
            p_pos = teammate_positions[i]
            d = np.sqrt((p_pos[0] - f_pos[0])**2 + (p_pos[1] - f_pos[1])**2)
            dists.append((i, d))
        
        dists.sort(key=lambda x: x[1])
        role_prefs[j] = [x[0] for x in dists]
    
    free = list(range(n))
    matches = {}
    for j in range(n):
        matches[j] = None
    
    proposal_idx = {}
    for i in range(n):
        proposal_idx[i] = 0
    
    while len(free) > 0:
        p = free[0]
        
        if proposal_idx[p] >= n:
            free.pop(0)
            continue
            
        r = player_prefs[p][proposal_idx[p]]
        proposal_idx[p] += 1
        
        if matches[r] is None:
            matches[r] = p
            free.pop(0)
        else:
            current = matches[r]
            current_rank = role_prefs[r].index(current)
            new_rank = role_prefs[r].index(p)
            
            if new_rank < current_rank:
                matches[r] = p
                free.pop(0)
                free.append(current)
    
    point_preferences = {}
    for r in matches:
        p = matches[r]
        point_preferences[p + 1] = formation_positions[r]
    
    return point_preferences
