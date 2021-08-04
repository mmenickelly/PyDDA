from ..cost_functions import *
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

def auglag_function(winds,parameters,mult,mu,resto,x0=None):

    al = 0
    al_grad = np.zeros(len(winds))

    if not resto:
        Jvel, Jvel_grad = radial_velocity_function(winds, parameters)
        al += Jvel
        al_grad += Jvel_grad

    if parameters.Cb > 0 and not resto:
        winds = np.reshape(winds,(3,parameters.grid_shape[0],parameters.grid_shape[1],parameters.grid_shape[2]))
        al += calculate_background_cost(
                winds[0], winds[1], winds[2], parameters.bg_weights,
                parameters.u_back, parameters.v_back, parameters.Cb)
        al_grad += calculate_background_gradient(
            winds[0], winds[1], winds[2], parameters.bg_weights,
            parameters.u_back, parameters.v_back, parameters.Cb)
        winds = winds.flatten()
    
    if resto:
        mult = np.zeros(mult.shape)
        mu = 1.0

    if parameters.Cm > 0:
        Jmass, Jmass_grad = al_mass_cont_function(winds, parameters, mult, mu)
        al += Jmass
        al_grad += Jmass_grad

    return al, al_grad

class Filter:
    def __init__(self, cv0, g0, beta = 0.5, gamma = 0.5):
        self.cvs = np.array([cv0])
        self.gs = np.array([g0])
        self.cv_min = cv0
        self.g_min = g0
        self.beta = beta
        self.gamma = gamma

    def add_to_filter(self, cv, g, cvtol, gtol):
        self.cvs = np.append(self.cvs,cv)
        self.gs = np.append(self.gs,g)
        if cv < self.cv_min:
            self.cv_min = np.maximum(cvtol,cv)
        if g < self.g_min:
            self.g_min = np.maximum(gtol,g)

    def check_acceptable(self, cv, g):
        cond1 = (cv <= self.beta*self.cvs)
        cond2 = (g <= (self.gs - self.gamma*cv))
        acceptable = np.logical_or(cond1,cond2)

        return acceptable.all()

class StopOptimizingException(Exception):
    pass

class Callback:
    def __init__(self, al, g, obj_func,parameters,theta = 0.01):
        self.al = al
        self.g = g
        self.theta = theta
        self.obj_func = obj_func
        self.parameters = parameters
        self.gnew = g
        self.alnew = al
    def __call__(self,xk):
        alnew, al_grad = self.obj_func(xk,self.parameters)
        al_grad = np.reshape(al_grad, (3, self.parameters.grid_shape[0], self.parameters.grid_shape[1], self.parameters.grid_shape[2]))
        al_grad[2,-1,:,:] = 0
        al_grad[2,0,:,:] = 0
        gnew = np.linalg.norm(al_grad.flatten(),np.inf)
        self.gnew = gnew
        self.alnew = alnew
        if (alnew <= np.minimum(100,self.al - self.theta*self.g)) and gnew <= 0.01:
            self.winds = xk
            raise StopOptimizingException()
            return True
        else:
            return False
        #return False
class RestoCallback:
    def __init__(self, AL_Filter, obj_func, parameters):
        self.AL_Filter = AL_Filter
        self.obj_func = obj_func
        self.parameters = parameters

    def __call__(self,xk):
        alnew, al_grad = self.obj_func(xk,self.parameters)
        g = np.linalg.norm(al_grad,np.inf)
        winds = np.reshape(xk, (3, self.parameters.grid_shape[0], self.parameters.grid_shape[1], self.parameters.grid_shape[2]))
        div = calculate_mass_continuity(winds[0],winds[1],winds[2],self.parameters.z,self.parameters.dx,self.parameters.dy,self.parameters.dz)
        cv = np.linalg.norm(div.flatten(),np.inf)
        self.cv = cv
        self.g = g
        self.al = alnew
        if self.AL_Filter.check_acceptable(cv,g):
            self.winds = xk
            raise StopOptimizingException()
            return True
        else:
            return False

def auglag(winds,parameters,bounds):

    mu = parameters.Cm
    
    # stopping criteria:
    cvtol = parameters.cvtol # maximum constraint violation must be less than this number (abs value of largest divergence must be less than this many 1/s units)
    gtol = parameters.gtol # Augmented Lagrangian norm must be less than this number
    Jveltol = parameters.Jveltol # acceptable terminating value of Jvel

    n = len(winds)

    # generate a random initial point
    #winds = np.random.normal(size=winds.shape)
    winds = np.reshape(winds, (3, parameters.grid_shape[0], parameters.grid_shape[1], parameters.grid_shape[2]))
    # ensure initial point satisfies impermeability condition:
    winds[2,-1,:,:] = 0
    winds[2,0,:,:] = 0
    
    # generate a (very coarse) guess of Lagrange multipliers
    div = calculate_mass_continuity(winds[0],winds[1],winds[2],parameters.z,parameters.dx,parameters.dy,parameters.dz)
    mult = -mu*div.flatten()
    zeromult = np.zeros(len(mult)) 
    
    winds = winds.flatten()
    
    cv0 = np.linalg.norm(div.flatten(),np.inf)
    print("Initial maximum constraint violation: ", "{:.6f}".format(cv0))
    print("Number of divergence constraints: ", len(mult))
    
    # initialize filter
    resto = False
    obj_func = lambda winds, parameters: auglag_function(winds, parameters, mult, mu, resto)
    al, al_grad = obj_func(winds, parameters)
    g = np.linalg.norm(al_grad,np.inf)
    print("Initial Augmented Lagrangian norm: ", "{:.6f}".format(g))
    AL_Filter = Filter(cv0,g)
    
    iter_count = 1
    while True:
        while True:
            # run L-BFGS-B with current fixed values of mult and mu 
            resto = False
            obj_func = lambda winds, parameters: auglag_function(winds, parameters, mult, mu, resto)
            cb = Callback(al,g,obj_func,parameters)
            try:
                winds = fmin_l_bfgs_b(obj_func, winds, args=(parameters,), callback=cb, maxiter = 20, bounds=bounds, approx_grad=False, disp=1,iprint=-1)
                winds = np.reshape(winds[0], (3, parameters.grid_shape[0], parameters.grid_shape[1], parameters.grid_shape[2]))
            except StopOptimizingException:
                winds = cb.winds
                winds = np.reshape(winds, (3, parameters.grid_shape[0], parameters.grid_shape[1], parameters.grid_shape[2]))
            
            g = cb.gnew
            al = cb.alnew
            # compute constraint violation and Lagrangian stationary measure:
            cv = 0.0
        
            if parameters.Cm > 0:
                div = calculate_mass_continuity(winds[0],winds[1],winds[2],parameters.z,parameters.dx,parameters.dy,parameters.dz)
                div = div.flatten()
                cv += np.linalg.norm(div,np.inf)

            # check if restoration is necessary:
            if AL_Filter.beta*np.maximum(AL_Filter.g_min/AL_Filter.gamma,AL_Filter.beta*AL_Filter.cv_min) <= cv or (g <= gtol and cv >= AL_Filter.beta*AL_Filter.cv_min):
                resto = True
                # increase penalty
                mu = 2.0*mu
                # run L-BFGS-B to minimize constraint violation
                print("Restoration phase, mu = :", mu)
                obj_func = lambda winds, parameters: auglag_function(winds, parameters, mult, mu, False)
                obj_func_resto = lambda winds, parameters: auglag_function(winds, parameters, zeromult, 1.0, resto, x0 = winds.flatten())
                resto_cb = RestoCallback(AL_Filter,obj_func,parameters)
                try:
                    winds = fmin_l_bfgs_b(obj_func_resto, winds, args=(parameters,), pgtol=0, maxiter=20, callback=resto_cb, bounds=bounds, approx_grad=False, disp=1,iprint=-1)
                    winds = np.reshape(winds[0], (3, parameters.grid_shape[0], parameters.grid_shape[1], parameters.grid_shape[2]))
                except StopOptimizingException:
                    winds = resto_cb.winds
                    winds = np.reshape(winds, (3, parameters.grid_shape[0], parameters.grid_shape[1], parameters.grid_shape[2]))
                try:
                    cv = resto_cb.cv
                    g = resto_cb.g
                    al = resto_cb.al
                except:
                    print("Can't make progress in restoration, ending prematurely")
                    return winds, mult
            else:
                # update multipliers
                mult = mult - mu*div
        
            # print some progress stats
            Jvel, Jvelgrad = radial_velocity_function(winds, parameters)
            print('Iter: ',iter_count)
            iter_count += 1
            print('Jvel: ',Jvel)
            print('Maximum constraint violation: ', cv)
            print("Augmented Lagrangian norm: ", "{:.6f}".format(g))

            # check if acceptable to filter
            if AL_Filter.check_acceptable(cv,g) or (cv <= cvtol and g <= gtol) or (cv <= cvtol and Jvel <= Jveltol):
                break

        # check stopping criteria
        if (cv <= cvtol and g <= gtol) or (cv <= cvtol and Jvel <= Jveltol):
            print('AugLag converged to specified tolerance')
            np.savetxt("cv.csv",div,delimiter=",")
            break

        # add newest point to filter
        AL_Filter.add_to_filter(cv,g,cvtol,gtol)

    return winds, mult






