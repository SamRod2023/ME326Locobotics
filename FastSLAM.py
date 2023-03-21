import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance


def RobotTrajectory(t):

    # Synthetic robot trajectory

    radius = 2
    T = 15
    x = radius * np.cos( 2*np.pi*t/T )
    y = radius * np.sin( 2*np.pi*t/T )

    return x, y

def search_blocks( M, px, py, blocks ):

    # Obtain measurement of blocks within range

    robot_range = 2
    _, n = M.shape
    p = np.array([px, py]).reshape(2,)
    X = np.empty((2, 0))
    colors = []
    for block in range(n):
        if np.linalg.norm( M[:, block] - p) <= robot_range:
            if block <= n/4 - 1 and blocks[0] == 1:
                X = np.concatenate((X, M[:,block].reshape(2,1)), axis=1 ) 
                colors.append('r')
            elif block >= n/4 and block <= n/2 -1 and blocks[1] == 1:
                X = np.concatenate((X, M[:,block].reshape(2,1)), axis=1 )
                colors.append('g')
            elif block >= n/2 and block <= 3*n/4 -1 and blocks[2] == 1:
                X = np.concatenate((X, M[:,block].reshape(2,1)), axis=1 )
                colors.append('b')
            elif block >= 3*n/4 and blocks[3]:
                X = np.concatenate((X, M[:,block].reshape(2,1)), axis=1 )
                colors.append('y')

    return X, colors

def update_seen_blocks(X, colors, mu, Sigma, seen_colors, noise_level):

    # Check if the block has already been seen based on l2 norm, if not add it to the list.

    m, n =  X.shape
    conf_interval = 0.95
    K2 = np.sqrt(-2*np.log(1-conf_interval))
    if mu.size == 0:
        for block in range(n):
            mu = np.append(mu, X[0, block])
            mu = np.append(mu, X[1, block])
        Sigma = noise_level*np.identity(2*n)
        seen_colors = colors
    else:
        r_init = int(mu.size/2)
        for block in range(n):
            add = True
            for i in range(r_init):
                sig = Sigma[2*i:2*i+2, 2*i:2*i+2]
                if np.linalg.norm(X[:,block].T - mu[2*i:2*i+2].T) <= 3*np.sqrt(noise_level):
                    if seen_colors[i] == colors[block]:
                        add = False
            if add:
                r = mu.size
                mu = np.append(mu, X[0, block])
                mu = np.append(mu, X[1, block])
                Sigma = np.block([[Sigma, np.zeros((r,2))], [np.zeros((2, r)), noise_level*np.identity(2)]])
                seen_colors.append(colors[block])

    return mu, Sigma, seen_colors

def extract_mu_sigma(X, color, mu, Sigma, seen_colors):

    # Associate seen measurement with corresponding block based on Mahalanobis distance

    r = mu.size
    closest_distance = np.inf

    for i in range(int(r/2)):
        sig = Sigma[2*i:2*i+2, 2*i:2*i+2]
        d = distance.mahalanobis(X.T, mu[2*i:2*i+2].T, sig)
        if d < closest_distance and color == seen_colors[i]:
            M = mu[2*i:2*i+2]
            closest_distance = d
            idx = 2*i

    return M, sig, idx

def measurement(M, p, theta):

    # Range measurement
    r = np.linalg.norm( M - p )
    # Bearing measurement
    d = M - p
    phi = np.arctan2(d[1], d[0]) - theta
    y_hat = np.array([r, phi])

    return y_hat

def UpdateFastSlam( mu, Sigma, X, colors, px, py, noise_level, blocks):

    theta = 0   # Change for the orientation of the robot
    m, n =  X.shape
    p = np.array([px, py]).reshape(2,)
    R = noise_level*np.identity(2)
    for block in range(n):
        color = colors[block]
        update = False
        if color == 'r' and blocks[0] == 1: update = True
        if color == 'g' and blocks[1] == 1: update = True
        if color == 'b' and blocks[2] == 1: update = True
        if color == 'y' and blocks[3] == 1: update = True
        if update:
            # Since A is the identity matrix, no need for predict step
            M, sigma, idx = extract_mu_sigma(X[:,block], color, mu, Sigma, seen_colors)
            C = MeasurementMatrix( M, p )
            # Update step
            K = np.matmul( sigma, np.matmul(C.T, np.linalg.inv( np.matmul( C, np.matmul(sigma, C.T) ) + R )))
            y_hat = measurement( M, p, theta )  # Estimated
            y = measurement( X[:,block], p, theta ) + 0.5*np.matmul( np.sqrt(R), np.random.randn(2) )
            inn =  y - y_hat
            meas = y + y_hat
            if np.abs(inn[1] > np.pi):
                inn[1] = meas[1]
                inn[1] = np.maximum( np.minimum(inn[1], np.pi), -np.pi )
            mu[idx:idx+2] = M + np.matmul( K, inn)
            Sigma[idx:idx+2, idx:idx+2] = sigma - np.matmul( K, np.matmul(C, sigma))

    return mu, Sigma

def MeasurementMatrix( M, p ):

    # Linearized measurement model

    d = M-p
    norm = np.linalg.norm(M-p)
    C = np.matrix([[d[0]/norm, d[1]/norm], [-d[1]/norm**2, d[0]/norm**2]])

    return C

# Assumption: The pose of the robot is known
# Assumption: The robot gathers both range and bearing measurements

blocks = [1, 0, 1, 0]       # Blocks to track (R,G,B,Y)
r = 0                       # State dimension 2x#-Features
rate = 8                    # Update rate
dt = 1/rate                 # Time-step

noise_level = 0.05
mu = np.empty((2,0))        # Prior r
Sigma = np.empty((0,0))     # Prior covariance rxr
seen_colors = []            # Seen colors
t = 0                       # Initial time
t_sim = 30                  # Simulation time

rng = np.random.default_rng(seed = 326)
blocks_in_map = 20
map_sz = 4
M = 2*map_sz*rng.random((2, blocks_in_map)) - map_sz

###
p = 20
theta = np.linspace( 0 , 2 * np.pi , p )
Rad = 0.5                                           # Locobot radius
conf_interval = 0.95
K2 = -2*np.log(1-conf_interval)
###

while t < t_sim:
    
    px, py = RobotTrajectory(t)                                                                     # Get position from Motion Capture System
    X, colors = search_blocks( M, px, py, blocks )                                                  # Replace by appropiate block detection function
    mu, Sigma, seen_colors = update_seen_blocks(X, colors, mu, Sigma, seen_colors, noise_level)
    mu, Sigma = UpdateFastSlam(mu, Sigma, X, colors, px, py, noise_level, blocks)
    t += dt

    # Plot results 

    fig = plt.figure(1)
    plt.subplot(1,2,1)
    ax = plt.gca()
    ax.set_aspect(1)
    for block in range(blocks_in_map):
        if block <= blocks_in_map/4 - 1:
            plt.plot(M[0, block], M[1, block], 'rs', markersize=5)
        elif block >= blocks_in_map/4 and block <= blocks_in_map/2 -1:
            plt.plot(M[0, block], M[1, block], 'gs', markersize=5)
        elif block >= blocks_in_map/2 and block <= 3*blocks_in_map/4 -1:
            plt.plot(M[0, block], M[1, block], 'bs', markersize=5)
        elif block >= 3*blocks_in_map/4:
            plt.plot(M[0, block], M[1, block], 'ys', markersize=5)
    plt.xlim([-map_sz, map_sz]); plt.ylim([-map_sz, map_sz])

    plt.subplot(1,2,2)
    plt.cla()
    ax = plt.gca()
    ax.set_aspect(1)
    plt.plot(px + Rad*np.cos(theta), py + Rad*np.sin(theta), 'k-', label='Locobot')
    for block in range(int(mu.size/2)):
        mu_i = mu[2*block:2*block+2]
        lam, V = np.linalg.eig(K2*Sigma[2*block:2*block+2, 2*block:2*block+2])
        lam = np.sqrt(lam)
        ellipse = np.zeros((2, p))
        for i in range(p):
            D = np.array([lam[0]*np.cos(theta[i]), lam[1]*np.sin(theta[i])]).reshape(2,1)
            ellipse[:, i] = (np.matmul(V, D) + mu_i.reshape(2,1)).reshape(2,)
        plt.fill(ellipse[0,:], ellipse[1,:], color=seen_colors[block], alpha=0.1, linestyle = 'None')
        plt.plot(mu_i[0], mu_i[1], color=seen_colors[block], marker='s', markersize=5)
        plt.xlim([-map_sz, map_sz]); plt.ylim([-map_sz, map_sz]); plt.legend(loc='upper left')
    plt.pause(dt)



