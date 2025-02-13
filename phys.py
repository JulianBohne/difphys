import raylibpy as rl
import torch
from torch.nn.functional import normalize
import math

particle_radius = 1.0
particle_mass = 1.0

base_stiffness = 1000
stiffness = 20
mouse_stiffness = 1

n = 5
particles = None
velocities = None
fixed = None
connections = None
rest_lengths = None

# Get squared distance in xs
# connections * ((torch.ones_like(connections) * particles[:,0]).T - particles[:,0])**2

# Get connected distances
# connections * (((particles - (torch.ones_like(connections).unsqueeze(dim=2) * particles).transpose(0, 1))**2).sum(dim=2)).sqrt()

def to_connection_matrix(vec):
    n = round((1 + math.sqrt(1 + 4*2*vec.shape[0])) / 2)
    assert vec.shape == ((n*(n - 1))//2,), "The number of elements in the vector does not match the number of elements in an upper triangular matrix without the diagonal."
    indices = torch.triu_indices(n, n, 1)
    connections = torch.zeros((n, n), dtype=vec.dtype)
    connections[indices[0], indices[1]] = vec
    return connections.T + connections

def setup_particles():
    global particles
    global velocities
    global connections
    global fixed
    global rest_lengths

    # n particles -> nx2 matrix
    particles = (torch.rand((n, 2), dtype=torch.float32) - 0.5) * 10# torch.tensor([[0, 0], [0, -2], [0, 2], [2, 0], [5, 0]], dtype=torch.float32)
    particles[0:3] = torch.tensor([[0, 0], [0, -2], [0, 2]], dtype=torch.float32)
    velocities = torch.zeros_like(particles)
    fixed = torch.zeros(particles.shape[0], dtype=torch.float32)
    fixed[0] = 1 # this one is the mouse
    fixed[1] = 1
    fixed[2] = 1
    possible_connection_count = (n*(n - 1)) // 2
    connection_amount = 0.25
    connections = to_connection_matrix(torch.floor(torch.rand((possible_connection_count,), dtype=torch.float32) + connection_amount)) * stiffness
    connections[0,:] = 0
    connections[:,0] = 0
    initial_particle_diffs = particles - (torch.ones_like(connections).unsqueeze(dim=2) * particles).transpose(0, 1)
    rest_lengths = (initial_particle_diffs**2).sum(dim=2).sqrt()

setup_particles()

# n particles -> nxn matrix 
# connections = torch.tensor([[0, stiffness, stiffness], [stiffness, 0, stiffness], [stiffness, stiffness, 0]], dtype=torch.float32)

# rest_lengths = torch.tensor([[0, 2, 0], [2, 0, 0], [0, 0, 0]], dtype=torch.float32)

gravity = torch.tensor([0, -9.81], dtype=torch.float32)

dt = 0.1 * 1.0/60.0

def do_physics():
    global particles
    global velocities
    
    acc_time = 0.0

    for _ in range(round((1.0/60.0) / dt)):
        acceleration = torch.zeros_like(velocities)

        particle_diffs = particles - (torch.ones_like(connections).unsqueeze(dim=2) * particles).transpose(0, 1)
        directions = normalize(particle_diffs, dim=2)
        particle_distances = (particle_diffs**2).sum(dim=2).sqrt()
        move_amount = connections * base_stiffness * (particle_distances - rest_lengths)
        movement = (directions.transpose(0, 2) * move_amount).transpose(0, 2).sum(dim=1)
        acceleration += movement;

        acceleration += gravity # Add outside forces
        acceleration = (acceleration.T * (1 - fixed)).T # Fix some points in place

        velocities += acceleration * dt

        particles += velocities * dt
        
        acc_time += dt

rl.set_config_flags(rl.FLAG_MSAA_4X_HINT)
rl.init_window(800, 600, "Weee wacky physics")
rl.set_target_fps(60)

def w2s(p):
    return p[0]*50, -p[1]*50

def s2w(p):
    return (p[0] - rl.get_render_width()/2)/50, -(p[1] - rl.get_render_height()/2)/50

grab_index = 0

while not rl.window_should_close():

    if rl.is_key_pressed(rl.KEY_R):
        setup_particles()
    elif rl.is_key_pressed(rl.KEY_UP):
        n += 1
        setup_particles()
    elif rl.is_key_pressed(rl.KEY_DOWN):
        n -= 1
        if n < 3:
            n = 3
        setup_particles()

    rl.begin_drawing()

    rl.rl_translatef(rl.get_render_width()/2, rl.get_render_height()/2, 0)
    
    mouse = torch.tensor(s2w(rl.get_mouse_position()), dtype=torch.float32)

    particles[0] = mouse

    if (rl.is_mouse_button_pressed(rl.MOUSE_BUTTON_LEFT)):
        grab_index = torch.argmin(((particles[1:] - mouse) ** 2).sum(dim = 1)).item() + 1
        connections[0, grab_index] = mouse_stiffness
        rest_lengths[0, grab_index] = 0
        connections[grab_index, 0] = mouse_stiffness
        rest_lengths[0, grab_index] = 0
    elif rl.is_mouse_button_released(rl.MOUSE_BUTTON_LEFT):
        connections[0, grab_index] = 0
        connections[grab_index, 0] = 0
        grab_index = 0

    do_physics()
    velocities *= 0.95
    
    rl.clear_background(rl.DARKGRAY)  # Clear the screen with a dark gray color

    n = particles.shape[0]

    for i in range(n):
        a_x, a_y = w2s(particles[i])
        for j in range(i+1, n):
            if connections[i][j] > 0:
                b_x, b_y = w2s(particles[j])
                rl.draw_line(a_x, a_y, b_x, b_y, rl.WHITE)

    for i in range(1, n):
        rl.draw_circle(particles[i,0]*50, -particles[i,1]*50, 5, rl.RED)
    
    rl.draw_fps(-rl.get_render_width()/2 + 10, -rl.get_render_height()/2 + 10)

    rl.end_drawing()


rl.close_window()
