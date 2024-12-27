import math
from main.structs.meshes.base_mesh import BaseMesh

def translation(vx, vy, dt):
    velocity = lambda p : [dt*vx, dt*vy]
    return velocity

# Rotation counterclockwise about center at 1 radian per second
def rotation(center, dt):
    velocity = lambda p : [-dt*(p[1]-center[1]), dt*(p[0]-center[0])]
    return velocity

# Vortex example from Rider and Kothe
def vortex(dt, time_reverse):
    velocity = lambda p : [-2*dt*(math.sin(math.pi*p[0]))**2 * math.sin(math.pi*p[1]) * math.cos(math.pi*p[1]), 2*dt*math.sin(math.pi*p[0])*math.cos(math.pi*p[0]) * (math.sin(math.pi*p[1]))**2]
    return velocity
    
# Reverses at time time_reverse (after time_reverse/dt time steps)
def vortex_tr(dt, t, totalt):
    velocity = lambda p : [-2*dt*math.cos(math.pi*t/totalt)*(math.sin(math.pi*p[0]))**2 * math.sin(math.pi*p[1]) * math.cos(math.pi*p[1]), 2*dt*math.cos(math.pi*t/totalt)*math.sin(math.pi*p[0])*math.cos(math.pi*p[0]) * (math.sin(math.pi*p[1]))**2]
    return velocity

# Initialize velocity for advection tests
def initializeVelocity(m: BaseMesh, t_total, test_setting="vortex"):
    if test_setting == "vortex":
        velocity = lambda t, p : [-200*math.cos(math.pi*t/t_total)*(math.sin(math.pi*p[0]/100))**2 * math.sin(math.pi*p[1]/100) * math.cos(math.pi*p[1]/100),
                                200*math.cos(math.pi*t/t_total)*math.sin(math.pi*p[0]/100)*math.cos(math.pi*p[0]/100) * (math.sin(math.pi*p[1]/100))**2]
    if test_setting == "deformation":
        velocity = lambda t, p: [100*math.cos(math.pi*t/t_total)*math.sin(4*math.pi*(p[0]+50)/100)*math.sin(4*math.pi*(p[1]+50)/100),
                                100*math.cos(math.pi*t/t_total)*math.cos(4*math.pi*(p[0]+50)/100)*math.cos(4*math.pi*(p[1]+50)/100)]
    elif test_setting == "zalesak":
        velocity = lambda t, p: [(2*math.pi)/t_total*(50-p[1]), (2*math.pi)/t_total*(p[0]-50)]
    elif test_setting == "xpluso":
        velocity = lambda t, p: [6, 6]
    elif test_setting == "triangle":
        velocity = lambda t, p: [6, 6]
    elif test_setting == "rectangle":
        velocity = lambda t, p: [6, 6]

    return velocity
