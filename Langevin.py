import numpy as np
from Potential import *

class Langevin2D():
    def __init__(self, square_edge, dt, friction, diffusion, forces, initial_position=np.zeros((2,))):
        self.gamma  = friction
        self.einstD = diffusion
        self.L      = square_edge
        self.forces = forces

        self.pos = initial_position
        self.step = 0
        self.dt = dt

    def do(self):
        pos = self.pos
        # Computing deterministic force
        f_deterministic = self.forces[0].evalForce(pos) 
        for i in range(1, len(self.forces)):
            f_deterministic += self.forces[i].evalForce(pos)
        f_sto = np.random.normal(size=(2,))
        # Computing euler increment
        self.pos = pos + f_deterministic/self.gamma*self.dt + self.einstD*f_sto*np.sqrt(self.dt); 
        # Applying PBC
        self.pos[0] = self.pos[0] % self.L
        self.pos[1] = self.pos[1] % self.L  
        # Updating time and step
        self.step += 1;

        return self.pos
    
    def refresh(self, start_position):
        self.pos = start_position;
        self.step = 0;

# integrates n_points of motion 
def langevinRun(langevin, n_points, t_start, step=1, write_file=None):
    n=0;
    if write_file != None:
        write_file.write(f'{t_start} {langevin.pos[0]} {langevin.pos[1]}')
    else:
        time_vec = [t_start];
        pos_vec = [langevin.pos];

    while n < n_points:
        langevin.do();
        n += 1;
        if n % step == 0:
            if write_file == None:
                time_vec.append(time_vec[-1]+langevin.dt);
                pos_vec.append(langevin.pos);
            else:
                write_file.write(f'{time_vec[-1]} {langevin.pos[0]} {langevin.pos[1]}')
    if write_file != None:
        write_file.close()
    return np.array(time_vec), np.array(pos_vec)
