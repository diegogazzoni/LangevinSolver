import numpy as np

class PotFunct():
    def __init__(self):
        self.P = 0
        self.F = np.zeros((2))
    
    def evalPotential(self, X):
        pass
    
    def evalForce(self, X):
        pass

class LinearPotential(PotFunct):
    def __init__(self, kx, ky):
        super().__init__()    
        self.kx = kx;
        self.ky = ky;

    def evalPotential(self, X):
        self.P = self.kx*X[0]+ self.ky*X[1];
        return self.P
    
    def evalForce(self, X):
        self.F = -np.array( [ (self.kx*X[0]**2)/2, (self.ky*X[1]**2)/2] )
        return self.F    
    
class GaussianPotential(PotFunct):
    def __init__(self, weights_mins, locations_mins, amplitudes):
        super().__init__()
        self.w_mins = weights_mins;
        self.centers = locations_mins;
        self.amp_factors = amplitudes;
    
    def __gauss2D(self, x, y, x_c, y_c, s_x=1, s_y=1):
        return np.exp( -( 0.5*((x-x_c)**2)/s_x**2 + 0.5*((y-y_c)**2)/s_y**2) )    

    def __grad_gauss2D(self, x, y, x_c, y_c, s_x, s_y):
        Dx = -(x-x_c)/(s_x**2) * self.__gauss2D(x,y,x_c,y_c,s_x,s_y);
        Dy = -(y-y_c)/(s_y**2) * self.__gauss2D(x,y,x_c,y_c,s_x,s_y);
        return np.array([Dx, Dy]);

    def evalPotential(self, X):
        x = X[0]
        y = X[1]
        self.P = self.w_mins * self.__gauss2D(x,y,self.centers[0],self.centers[1], self.amp_factors, self.amp_factors) 
        return self.P

    def evalForce(self, X):
        x = X[0]
        y = X[1]
        self.F = -self.w_mins * self.__grad_gauss2D(x, y, self.centers[0],self.centers[1], self.amp_factors, self.amp_factors) 
        return self.F    
