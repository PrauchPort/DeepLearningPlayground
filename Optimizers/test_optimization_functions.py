import autograd.numpy as np
from autograd import grad



class Beale:
    def __init__(self):
        self.xmin, self.xmax = -1.5, 5
        self.ymin, self.ymax = -3, 1.5
        self.y_start, self.x_start = -2.8, 2.4  # Start point
        self.x_optimum, self.y_optimum, self.z_optimum = 3, 0.5, 0  # Global optimum 
        self._compute_derivatives()
        
    def _compute_derivatives(self):
        self.df_dx = grad(self.eval, 0) # Partial derivative of the objective function over x
        self.df_dy = grad(self.eval, 1) # Partial derivative of the objective function over y
        
    def eval(self, x, y):
        z = np.log(1+(1.5-x+x*y)**2 + (2.25-x+x*y**2)**2 + (2.625-x+x*y**3)**2)/10
        return z
    
    
    
    
    
    
    
    
    
    
    
    
class Booth:
    def __init__(self):
        self.x_min, self.x_max = -5, 7
        self.y_min, self.y_max = -5, 7
        self.start_x, self.start_y = -4.1, 4.3
        self.x_optimum, self.y_optimum, self.z_optimum = 1, 3, 0
        self._compute_derivatives()
        
    def _compute_derivatives(self):
        self.dx_dz = grad(self.eval, 0)
        self.dy_dz = grad(self.eval, 1)
        
    def eval(self, x, y):
        # minimum at (1, 3)
        z = (x + 2*y - 7)**2 +(2*x + y - 5)**2
        return z