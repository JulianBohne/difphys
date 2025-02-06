import raylibpy as rl
import torch
from torch.nn.functional import normalize

particle_radius = 1.0
particle_mass = 1.0
# n particles -> nx2 matrix
particles = torch.tensor([[1, 0], [3, 0], [3, 1]], dtype=torch.float64)
velocities = torch.zeros_like(particles)
may_move = torch.tensor([0, 1, 1], dtype=torch.float64)

# Get squared distance in xs
# connections * ((torch.ones_like(connections) * particles[:,0]).T - particles[:,0])**2

# Get connected distances
# connections * (((particles - (torch.ones_like(connections).unsqueeze(dim=2) * particles).transpose(0, 1))**2).sum(dim=2)).sqrt()

stiffness = 500

# n particles -> nxn matrix 
connections = torch.tensor([[0, stiffness, stiffness], [stiffness, 0, stiffness], [stiffness, stiffness, 0]], dtype=torch.float64)
initial_particle_diffs = particles - (torch.ones_like(connections).unsqueeze(dim=2) * particles).transpose(0, 1)
rest_lengths = (initial_particle_diffs**2).sum(dim=2).sqrt()
# rest_lengths = torch.tensor([[0, 2, 0], [2, 0, 0], [0, 0, 0]], dtype=torch.float64)

gravity = torch.tensor([0, -9.81], dtype=torch.float64)

dt = 0.1 * 1.0/60.0

def do_physics():
    global particles
    global velocities
    
    acc_time = 0.0

    for _ in range(round((1.0/60.0) / dt)):

        # Find distances to other particles, n particles -> nxn matrix
        acceleration = torch.zeros_like(velocities)

        particle_diffs = particles - (torch.ones_like(connections).unsqueeze(dim=2) * particles).transpose(0, 1)
        directions = normalize(particle_diffs, dim=2)
        particle_distances = (particle_diffs**2).sum(dim=2).sqrt()
        move_amount = connections * (particle_distances - rest_lengths)
        movement = (directions.transpose(0, 2) * move_amount).transpose(0, 2).sum(dim=1)
        acceleration += movement;

        acceleration += gravity # Add outside forces
        acceleration = (acceleration.T * may_move).T # Fix some points in place

        velocities += acceleration * dt
        # velocities *= 0.95

        particles += velocities * dt
        
        acc_time += dt

rl.init_window(800, 600, "Weee wacky physics")
rl.set_target_fps(60)

def w2s(p):
    return p[0]*50, -p[1]*50

while not rl.window_should_close():
    x = rl.get_mouse_x()
    y = rl.get_mouse_y()

    rl.begin_drawing()

    rl.rl_translatef(rl.get_render_width()/2, rl.get_render_height()/2, 0)
    
    rl.clear_background(rl.DARKGRAY)  # Clear the screen with a dark gray color

    n = particles.shape[0]

    for i in range(n):
        a_x, a_y = w2s(particles[i])
        for j in range(n):
            if connections[i][j] > 0:
                b_x, b_y = w2s(particles[j])
                rl.draw_line(a_x, a_y, b_x, b_y, rl.WHITE)

    for i in range(n):
        rl.draw_circle(particles[i,0]*50, -particles[i,1]*50, 5, rl.RED)
    
    rl.end_drawing()

    do_physics()

rl.close_window()
