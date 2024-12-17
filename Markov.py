import numpy as np

class MarkovModel():
    def __init__(self, discrete_trajectories, lag_time):
        self.discrete_trajectories = discrete_trajectories;
        self.lag_time = lag_time;
        self.states, self.num_states, self.labelled_trajectories = self.__disc2label(discrete_trajectories);
        self.equilibrium_distribution = None
        self.T = None
        self.C = None
        self.timescales = None
        self.slowest_ev = None
    
    def __disc2label(self, discrete_trajectories):
        discrete_trajs_merged = np.vstack([tr for tr in discrete_trajectories])
        states, label_trajs, n_counts = np.unique(discrete_trajs_merged, axis=0, return_counts=True, return_inverse=True)
        lens = np.array([len(tr) for tr in discrete_trajectories])
        label_trajs = np.split(label_trajs, axis=0, indices_or_sections=lens.cumsum()[:-1])
        return states, len(states), label_trajs
        
    def __Tmatrix(self, label_trajs, disc_trajs, states, lag_time):
        n_states = len(states)
        C = np.zeros((n_states, n_states))
        for traj in label_trajs:
            for n in range(len(traj)-lag_time):
                C[traj[n], traj[n+lag_time]] += 1  
        rem_states = []
        new_states = states.copy()
        new_trajs = []
        new_trajs_discrete = []
        while True:
            C_rows = np.sum(C, axis=1); 
            id_0 = np.where(C_rows == 0)[0];
            if len(id_0) == len(rem_states):
                break
            for rm in id_0:
                C[rm, :] = 0; 
                C[:, rm] = 0;
                if rm not in rem_states:
                    rem_states.append(rm)
        C = np.delete(C, rem_states, axis=0)
        C = np.delete(C, rem_states, axis=1)
        #print(f'Deleted {rem_states}')

        new_states = np.delete(new_states, rem_states, axis=0)
        new_states_list = [s for s in range(n_states) if s not in rem_states]
        for traj, d_traj in zip(label_trajs, disc_trajs):
            new_traj = traj.copy();
            new_traj_d = d_traj.copy()
            for rmstate in rem_states:
                time_torem = np.where(new_traj == rmstate)[0];
                new_traj   = np.delete(new_traj, time_torem);
                new_traj_d = np.delete(new_traj_d, time_torem, axis=0)
            for oldstate in new_states_list:
                to_update = np.where(new_traj == oldstate);
                new_traj[to_update] = new_states_list.index(oldstate) # updating state index
            new_trajs.append(new_traj)
            new_trajs_discrete.append(new_traj_d)
        T = C / np.sum(C, axis=1).reshape((-1,1)).T 
        return C, T, new_states, new_trajs, new_trajs_discrete

    def makeModel(self, show_debug=False):
        self.C, self.T, self.states, self.labelled_trajectories, self.discrete_trajectories = self.__Tmatrix(self.labelled_trajectories, self.discrete_trajectories, self.states, self.lag_time) 
        
        # Matrix checking
        if np.sum(self.C, axis=1).all() == 0:
            print("Wrong count matrix produced!");
            return
        if show_debug:
            print(f'Pre-normalization count matrix: {np.sum(C, axis=1)}')
        
        ea, ev = np.linalg.eig(self.T);
        l = np.argsort(ea)[::-1];
        ea = ea[l];
        ev = ev[:, l];
        taus = -self.lag_time / np.log( np.abs(ea[1:]) );
        p_inf = np.abs( ev[:, 0] );
        p_inf /= np.sum(p_inf)
        slowest_ev = ev[:, 1]; 

        self.equilibrium_distribution = p_inf;
        self.timescales = taus;
        self.slowest_ev = slowest_ev;

        return self.T, self.states, self.labelled_trajectories, taus, slowest_ev, p_inf
    
def cont2discrete(trajs, L, dL):
    bin_edges = np.arange(0, L, dL);
    bin_centers = (bin_edges[1:] + bin_edges[:-1])*0.5
    discrete_trajs = []
    for traj in trajs:
        idx = np.digitize(traj[:, 0], bin_edges[:-1])-1;
        idy = np.digitize(traj[:, 1], bin_edges[:-1])-1;
        discrete_trajs.append( np.vstack( (idx, idy) ).T )        
    return bin_edges, bin_centers, discrete_trajs

