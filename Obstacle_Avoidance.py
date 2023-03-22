import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.optimize import fsolve
import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')

Obstacles = {'1': {'pos': np.array([-5,0]), 'rad': 0.25, 'color':'r'},
            '2': {'pos': np.array([5,0]), 'rad': 0.25, 'color':'g'},
            '3': {'pos': np.array([0,0]), 'rad': 0.25, 'color':'b'},
            '4': {'pos': np.array([-3,-1]), 'rad': 0.25, 'color':'y'},
            '5': {'pos': np.array([2,3]), 'rad': 0.25, 'color':'r'}}
Locobot = {'pos': np.array([-5, -5]),
           'vel': np.array([0, 0]),
           'tar': np.array([4.5,5]),
           'rad': 0.5}
Agent = {'pos': np.array([-5, 2.5]),
         'vel': np.array([0.25, -0.1]),
         'rad': 0.5}

def sys_equations(var, *data):
    x0, y0, a, b = data
    x, y = var
    eq1 = a**2*y*(y-y0) + b**2*x*(x-x0)
    eq2 = (x-x0)**2/a**2 + (y-y0)**2/b**2 - 1
    return [eq1, eq2]

def RVO(Locobot, Agent, Obstacles):

    '''
    Locobot:
        -> Locobot-pos: Position of the Locobot
        -> Locobot-vel: Velocity of the Locobot
        -> Locobot-tar: Target position of the Lovobot
    Agent (Other Locobot):
        -> Agent-pos: Position of the other Locobot
        -> Agent-vel: Velocity of the other Locobot
        -> Agent-pos_cov: Covariance matrix of the estimated position of the agent
        -> Agent-vel_cov: Covariance matrix of the estimated velocity of the agent
    Obstacles:
        -> Obstacle[i]-pos: Position of obstacle i
        -> Obstalce[i]-rad: Radius of obstacle i
        -> Obstacle[i]-pos_cov: Covariance matrix of the estimated position of obstacle i
    '''

    ####
    plt.cla()
    ####
    
    v_max = 0.3
    v_pre = v_max*((Locobot['tar']-Locobot['pos'])/np.linalg.norm(Locobot['pos']-Locobot['tar'])).reshape(2,1)# + 0.0001*np.random.randn(2,1)
    eps_safe = 0.25

    ####
    theta = np.linspace( 0 , 2 * np.pi , 80 )
    ####

    lines = np.empty((0, 2))
    obs_tag = []
    rad = 3

    # Static Obstacles
    if Obstacles.keys():
        for i in Obstacles.keys():
            x0, y0 = Obstacles[i]['pos'] - Locobot['pos']
            if x0**2+y0**2 <= rad**2:
                a = Obstacles[i]['rad'] + Locobot['rad'] + eps_safe; b = Obstacles[i]['rad'] + Locobot['rad'] + eps_safe
                x1, y1 = fsolve(sys_equations, [x0 + a, y0 - 0.05], args=(x0, y0, a, b))
                x2, y2 = fsolve(sys_equations, [x0 - a, y0 + 0.05], args=(x0, y0, a, b))
                unit_vec_right = np.array([-y1/np.sqrt(x1**2 + y1**2), x1/np.sqrt(x1**2 + y1**2)]).reshape(1,2)
                unit_vec_left = np.array([-y2/np.sqrt(x2**2 + y2**2), x2/np.sqrt(x2**2 + y2**2)]).reshape(1,2)
                lines = np.append(lines, unit_vec_left, 0)
                lines = np.append(lines, unit_vec_right, 0)
                obs_tag.append(i)

            ####
            plt.plot(Obstacles[i]['pos'][0], Obstacles[i]['pos'][1], color=Obstacles[i]['color'], marker='s', markersize = 5)
            ####

    if np.linalg.norm(Agent['vel']) > 0:
        # Agent
        x0, y0 = Agent['pos'] - Locobot['pos']
        a = Agent['rad'] + Locobot['rad'] + eps_safe; b = Agent['rad'] + Locobot['rad'] + eps_safe
        x1, y1 = fsolve(sys_equations, [x0 + a, y0 - 0.05], args=(x0, y0, a, b))
        x2, y2 = fsolve(sys_equations, [x0 - a, y0 + 0.01], args=(x0, y0, a, b))
        unit_vec_right = np.array([-y1/np.sqrt(x1**2 + y1**2), x1/np.sqrt(x1**2 + y1**2)]).reshape(1,2)
        unit_vec_left = np.array([-y2/np.sqrt(x2**2 + y2**2), x2/np.sqrt(x2**2 + y2**2)]).reshape(1,2)

        # Possible solutions
        m, _ = lines.shape
        A = np.zeros((4*m, 4*m))
        b = np.zeros((4*m, 1))
        for i in range(m):
            A[2*i, 2*i:2*i+2] = lines[i,:]
            A[2*i+1, 2*i:2*i+2] = unit_vec_right
            b[2*i+1] = np.dot(unit_vec_right, 0.5*(Agent['vel'] + Locobot['vel']))
            A[2*i+2*m, 2*i+2*m:2*i+2+2*m] = lines[i,:]
            A[2*i+1+2*m, 2*i+2*m:2*i+2+2*m] = unit_vec_left
            b[2*i+1+2*m] = np.dot(unit_vec_left, 0.5*(Agent['vel'] + Locobot['vel']))

        X = np.linalg.solve(A, b)
        x_solv = X[::2]
        y_solv = X[1::2]

        # Add prefered velocity
        x_solv = np.append(x_solv, v_pre[0])
        y_solv = np.append(y_solv, v_pre[1])

        # Find velocity that is outside the union of the reciprocal velocity obstacles
        best_diff = np.inf
        eps = 1e-5
        for j in range(x_solv.size):
            feasible = True
            point = np.array([x_solv[j], y_solv[j]]).reshape(2,1)
            for i in range(int(m/2)):
                left_hyperplane = np.dot(lines[2*i,:], point)
                right_hyperplane = np.dot(lines[2*i+1,:], point )
                norm_dir = (Obstacles[obs_tag[i]]['pos']-Locobot['pos'])/np.linalg.norm(Obstacles[obs_tag[i]]['pos']-Locobot['pos'])
                if (left_hyperplane < -eps) and (right_hyperplane > eps) and (np.dot(norm_dir,v_pre/v_max) > -0.5):
                    feasible = False 
            if feasible:
                curr_diff = np.linalg.norm(point - v_pre)
                if curr_diff < best_diff:
                    v_cmd = point
                    best_diff = curr_diff
        if best_diff == np.inf:
            v_cmd = np.zeros((2,))
    
    else:
        # Find velocity that is outside the union of the reciprocal velocity obstacles
        m, _ = lines.shape
        eps = 1e-5
        feasible = True
        point = np.array([v_pre[0], v_pre[1]]).reshape(2,1)
        for i in range(int(m/2)):
            left_hyperplane = np.dot(lines[2*i,:], point)
            right_hyperplane = np.dot(lines[2*i+1,:], point )
            norm_dir = (Obstacles[obs_tag[i]]['pos']-Locobot['pos'])/np.linalg.norm(Obstacles[obs_tag[i]]['pos']-Locobot['pos'])
            if (left_hyperplane < -eps) and (right_hyperplane > eps) and (np.dot(norm_dir,v_pre/v_max) > -0.5):
                feasible = False 
        if feasible:
            v_cmd = point
        else:
            best_diff = np.inf
            for i in range(m):
                feasible = True
                point = v_max*np.array([lines[i,1], -lines[i,0]]).reshape(2,1)
                for j in range(int(m/2)):
                    left_hyperplane = np.dot(lines[2*j,:], point )
                    right_hyperplane = np.dot(lines[2*j+1,:], point )
                    norm_dir = (Obstacles[obs_tag[j]]['pos']-Locobot['pos'])/np.linalg.norm(Obstacles[obs_tag[j]]['pos']-Locobot['pos'])
                    if (left_hyperplane < -eps) and (right_hyperplane > eps) and (np.dot(norm_dir,v_pre/v_max) > -0.5):
                        feasible = False 
                if feasible:
                    curr_diff = np.linalg.norm(point - v_pre)
                    if curr_diff < best_diff:
                        v_cmd = point
                        best_diff = curr_diff
                        curr_diff = np.linalg.norm(point - v_pre)
                    if curr_diff < best_diff:
                        v_cmd = point
                        best_diff = curr_diff
            if best_diff == np.inf:
                v_cmd = np.zeros((2,))

    ###
    dt = 0.1
    Locobot['pos'] = Locobot['pos'] + v_cmd.reshape(2,)*dt
    Locobot['vel'] = v_cmd.reshape(2,)
    Agent['pos'] = Agent['pos'] + Agent['vel']*dt   
    plt.plot(Locobot['pos'][0] + Locobot['rad']*np.cos(theta), Locobot['pos'][1] + Locobot['rad']*np.sin(theta), 'k-', label='Locobot')
    plt.plot(Agent['pos'][0] + Agent['rad']*np.cos(theta), Agent['pos'][1] + Agent['rad']*np.sin(theta), 'g-', label='Agent')
    plt.plot(Locobot['tar'][0], Locobot['tar'][1], 'kx', markersize = 10, label='Target')
    plt.xlim([-6, 6]); plt.ylim([-6, 6]); plt.legend(loc='upper left')
    plt.pause(dt)
    ###

while np.linalg.norm(Locobot['tar'] - Locobot['pos']) >= 0.75:
    RVO(Locobot, Agent, Obstacles)

