import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as sp
import casadi as ca

import pdb

def create_casadi_spline_function(x,y,smoothness=10):
    scipy_spline = sp.UnivariateSpline(x, y, s=smoothness)
    return ca.interpolant("LUT","bspline",[x],scipy_spline(x))

def create_casadi_rbf_function(x_vec,y_vec,num_basis=10):
    b = 0.1
    alpha = 37.4
    C = np.linspace(0, 1, num_basis)

    c = (y_vec-b).reshape(-1,1)

    A = np.empty((0,num_basis))
    for x in x_vec:
      phi = np.array([ca.exp(-alpha * (x - centre) ** 2) for centre in C]).reshape(1,-1)
      A = np.concatenate((A,phi),axis=0)

    w = np.linalg.pinv(A)@c

    x = ca.SX.sym('x')
    phi = ca.vertcat(*[ca.exp(-alpha * (x - c) ** 2) for c in C])
    y = phi.T @ w + b
    return ca.Function('rbf', [x], [y])

def create_casadi_poly_function(x,y,deg):
    coeffs = np.polyfit(x=x,y=y,deg=deg)
    s = ca.SX.sym('s')
    f_expr = 0
    l = len(coeffs)
    for i in range(l):
        f_expr += coeffs[i]*s**(l-i-1)
    return ca.Function('poly', [s], [f_expr])

class ProbabilisticPCA:
    """A class that learns a motion model using pPCA as suggested in:
    De Schutter J., Aertbeliën E.; Learning a Predictive Model of Human Gait for
    the Control of a Lower-limb Exoskeleton, IEEE RAS & EMBS International
    Conference on Biomedical Robotics and Biomechatronics (BioRob) (2014),
    Sao Paolo. pp. 520-525."""

    def __init__(self, training_data, phase_speed=1, dt=0.01, modes=5, name=None):

        self.training_data = training_data
        self.demo_number = len(self.training_data)
        self.dim_number = len(self.training_data[0])
        self.traj_length = len(self.training_data[0][0, :])
        self.s_vec = np.linspace(0,1,self.traj_length)

        # Number of modes
        self.modes = modes

        if name is None:
            self.name = "Nameless"
        else:
            self.name = name
            
    def train(self, approx_type='poly'):
        """Train the motion model using the pPCA algorithm."""
        # Get some values
        traj_length = self.traj_length
        training_data = self.training_data
        demo_number = self.demo_number
        modes = self.modes
        d = self.dim_number*traj_length

        # step 1: put all trials after each other
        sample_matrix = np.empty((0, d))
        for i in range(0, demo_number):
            demo = np.array([[x for joint in training_data[i] for x in joint]])
            sample_matrix = np.append(sample_matrix, demo, axis=0)

        # step 2: create sample covariance matrix
        sample_cov_matrix = np.cov(sample_matrix.T)

        # step 3: calculate eigenvectors/eigenvalues
        eigvals, eigvecs = np.linalg.eig(sample_cov_matrix)
        eigvals = np.sort(eigvals) # Documentation states that eigvals are not sorted, but they are storted. The sort / flip command change nothing (not for the 5 important modes at least)
        eigvals = np.flip(eigvals)

        # Keep only real part (Is this ok, imaginary part is very small...)
        eigvals = np.real(eigvals)
        eigvecs = np.real(eigvecs)
        
        plt.plot(eigvals[:])
        plt.yscale('log')

        # step 4: select principal values
        princ_vals = np.diag(eigvals[:modes])
        princ_vecs = eigvecs[:, :modes]

        # step 5: calculate H matrix, meas_noise and b (max. likelihood sol.)
        b_ml = np.mean(sample_matrix, axis=0)
        meas_noise = np.sqrt(1/(d-modes) * np.sum(eigvals[modes:]))
        if np.sum(eigvals[modes:]) <= 0:
            print("Retry training of PPCA model with less modes (default = 5).")
            
        # h_ml = np.dot(princ_vecs, np.sqrt((princ_vals-np.power(meas_noise, 2) * np.eye(len(princ_vals)))))
        h_ml = princ_vecs @ np.sqrt(princ_vals - meas_noise**2*np.eye(modes))

        self.h_ml = h_ml
        self.b_ml = b_ml
        self.meas_noise = meas_noise
        
        # Create casadi functions for approximation
        if approx_type == 'poly':
            self.H_b_poly_approximation()
        elif approx_type == 'spline':
            self.H_b_spline_approximation()
        elif approx_type == 'rbf':
            self.H_b_rbf_approximation()
        else:
            print('Approximation type not supported')
        
    def H_b_poly_approximation(self):
        nl = self.modes
        nz = self.dim_number
        d = self.traj_length
        s = ca.SX.sym('s')
        s_vec = self.s_vec
        
        H_ca = ca.vertcat()
        for i in range(nz):
            row = ca.horzcat(*[ create_casadi_poly_function(s_vec,self.h_ml[ (d*i) : (d*(i+1)) ,j],deg=5)(s) for j in range(nl) ]) # Polynomial function
            H_ca = ca.vertcat(H_ca,row)
        self.H_ca_function = ca.Function('H', [s], [H_ca])

        b_ca = ca.vertcat(*[ create_casadi_poly_function(s_vec,self.b_ml[ (d*i) : (d*(i+1))],deg=5)(s) for i in range(nz) ]) # Polynomial function
        self.b_ca_function = ca.Function('b', [s], [b_ca])
        
    def H_b_spline_approximation(self):
        nl = self.modes
        nz = self.dim_number
        d = self.traj_length
        s = ca.MX.sym('s')
        s_vec = self.s_vec
        
        H_ca = ca.vertcat()
        for i in range(nz):
            row = ca.horzcat(*[ create_casadi_spline_function(s_vec,self.h_ml[ (d*i) : (d*(i+1)) ,j],smoothness=100)(s) for j in range(nl) ]) # Polynomial function
            H_ca = ca.vertcat(H_ca,row)
        self.H_ca_function = ca.Function('H', [s], [H_ca])

        b_ca = ca.vertcat(*[ create_casadi_spline_function(s_vec,self.b_ml[ (d*i) : (d*(i+1))],smoothness=100)(s) for i in range(nz) ]) # Polynomial function
        self.b_ca_function = ca.Function('b', [s], [b_ca])

    def H_b_rbf_approximation(self):
        nl = self.modes
        nz = self.dim_number
        d = self.traj_length
        s = ca.MX.sym('s')
        s_vec = self.s_vec
        
        H_ca = ca.vertcat()
        for i in range(nz):
            row = ca.horzcat(*[ create_casadi_rbf_function(s_vec,self.h_ml[ (d*i) : (d*(i+1)) ,j], num_basis=15)(s) for j in range(nl) ])
            H_ca = ca.vertcat(H_ca,row)
        self.H_ca_function = ca.Function('H', [s], [H_ca])

        b_ca = ca.vertcat(*[ create_casadi_rbf_function(s_vec,self.b_ml[ (d*i) : (d*(i+1))], num_basis=15)(s) for i in range(nz) ])
        self.b_ca_function = ca.Function('b', [s], [b_ca])
        
    def plot_H_b_approximation(self):
        s_vec = self.s_vec
        d=self.traj_length
        
        for i in range(self.dim_number):
            for j in range(self.modes):
                plt.figure()
                plt.title(r'$H_{'+str(i)+','+str(j)+'}$')
                plt.plot(s_vec, self.h_ml[(d*i) : (d*(i+1)),j])
                plt.plot(s_vec, [self.H_ca_function(s).full()[i,j] for s in s_vec])
                # plt.plot(s_vec, [self.evaluate_basisfunctions(s)[0][i,j] for s in s_vec])
                plt.legend(['raw_data', 'approx','spline'])
                plt.show()       
        
        for i in range(self.dim_number):
            plt.figure()
            plt.title(r'$b_{'+str(i)+'}$')
            plt.plot(s_vec, self.b_ml[(d*i) : (d*(i+1))])
            plt.plot(s_vec, [self.b_ca_function(s).full()[i] for s in s_vec])
            # plt.plot(s_vec, [self.evaluate_basisfunctions(s)[1][i] for s in s_vec])
            plt.legend(['raw_data', 'approx', 'spline'])
            plt.show()
        


    def plot_model(self):
        
        mean_vec = np.empty((self.dim_number,0)) # Used for storing mean of trajectory
        std_vec = np.empty((self.dim_number,0)) # Used for storing standard deviation of trajectory
        
        for s in self.s_vec:
            H_eval = self.H_ca_function(s).full()
            b_eval = self.b_ca_function(s).full()
            
            cov = H_eval @ H_eval.T + self.meas_noise**2 * np.eye(self.dim_number)
            std = np.array([np.sqrt(cov[i,i]) for i in range(cov.shape[0])]).reshape(-1,1)
            
            mean_vec = np.concatenate((mean_vec,b_eval), axis=1)
            std_vec = np.concatenate((std_vec,std), axis=1)
            
        for i in range(self.dim_number):
            plt.figure()
            plt.xlabel('progress')
            plt.ylabel('$y_'+str(i)+'$')
            plt.plot(self.s_vec, mean_vec[i])
            plt.plot(self.s_vec, mean_vec[i]-std_vec[i], color='black')
            plt.plot(self.s_vec, mean_vec[i]+std_vec[i], color='black')
            plt.fill_between(self.s_vec, 
                mean_vec[i]-std_vec[i],
                mean_vec[i]+std_vec[i], color='#555555', alpha=0.2
            )
            plt.ylim(-0.1,1)
            plt.show()