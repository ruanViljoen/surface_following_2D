import numpy as np
import casadi as ca
import pickle
import time
import pandas as pd
from tqdm import tqdm
from pseudo_etasl import TaskContext, generate_solver, utils
import yaml
import sys

#########################
# Experiment parameters #
#########################
experiment_name = sys.argv[1]

print("")
print("")
print("Starting experiment: " + experiment_name)
print("")
print("")

with open('experiment_parameters/'+experiment_name+'.yaml', "r") as stream:
    try:
        experiment_paramters = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

model_type = experiment_paramters['model_type']

q_weights = experiment_paramters['q_weights']
q_init = experiment_paramters['q_init']

task_translation_wn = experiment_paramters['task_translation_wn']
task_translation_zeta = experiment_paramters['task_translation_zeta']
task_translation_hard = experiment_paramters['task_translation_hard']
task_translation_weight = experiment_paramters['task_translation_weight']
task_translation_tolerance = experiment_paramters['task_translation_tolerance']

task_orientation_wn = experiment_paramters['task_orientation_wn']
task_orientation_zeta = experiment_paramters['task_orientation_zeta']
task_orientation_hard = experiment_paramters['task_orientation_hard']
task_orientation_weight = experiment_paramters['task_orientation_weight']
task_orientation_tolerance = experiment_paramters['task_orientation_tolerance']

task_progress_speed_K = experiment_paramters['task_progress_speed_K']
task_progress_speed_hard = experiment_paramters['task_progress_speed_hard']
task_progress_speed_weight = experiment_paramters['task_progress_speed_weight']

velocity_limit = experiment_paramters['velocity_limit']
acceleration_limit = experiment_paramters['acceleration_limit']

s_dot_desired_target = experiment_paramters['s_dot_desired_target']

MPC_horizon_length = experiment_paramters['MPC_horizon_length']
ts = experiment_paramters['ts']
mu = experiment_paramters['mu']

MHE_horizon_length = experiment_paramters['MHE_horizon_length']
buffer_distance_tolerance = experiment_paramters['buffer_distance_tolerance']

freq = experiment_paramters['freq']
T = experiment_paramters['T']
 
##################################
# Estimator and helper functions #
##################################
from ppca_ruan import ProbabilisticPCA

def estimate_surface_approx_coefficients(f_surface, x):
  x_sym = ca.SX.sym('x')
  f = f_surface(x_sym)

  # Find quadratic approximation at x
  df = ca.Function('jacobian', [x_sym],[ca.jacobian(f,x_sym)])(x).full().flatten()[0]
  ddf = ca.Function('hessian', [x_sym],[ca.hessian(f,x_sym)[0]])(x).full().flatten()[0]
  
  a = f_surface(x).full().flatten()[0] - df*x + 0.5*ddf*x**2
  b = df - ddf*x
  c = 0.5*ddf

  return [a,b,c]

class MHE:
  
  def __init__(self, window_length, h, R, mu):
    self.z_total = []
    self.x_total = []
    self.i = 0
    self.window_length = window_length
    self.h = h
    self.nw = self.h.nnz_in()-1
    self.R = R
    self.mu = mu
    self.previous_w = 0
  
  def step(self,x,z):
    
    self.x_total += [x]
    self.z_total += [z]
    
    window_start = 0
    if self.i >= self.window_length-1:
      window_start = self.i-self.window_length+1
    
    x_window = self.x_total[window_start:(self.i+1)]
    z_window = self.z_total[window_start:(self.i+2)]
  
    w = ca.SX.sym('w',self.nw)

    err_window = ca.vertcat(*[z_window[i]-(h(x_window[i],w)) for i in range(len(z_window))])
    err = err_window.T @ np.linalg.inv(self.R) @ err_window
    # err += self.mu*(w-self.previous_w).T@(w-self.previous_w) # Add regularization of weight velocity
    err += self.mu*w.T@w # Add regularization of weight value

    opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.tol": 1e-5}
    nlp = {'x': w, 'f': err}
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    # Initial guess
    w0 = np.ones((self.nw))
    # Solve
    res = solver(x0=w0)

    # Extract optimal values
    w_opt = res['x'].full().flatten()
    
    self.previous_w = w_opt.reshape(-1,1)
    self.i += 1

    return w_opt, x_window
  
def generate_path(model,x,w,p):
  x_sym = ca.MX.sym('x')
  f_expr = model(x_sym,w)

  m_expr = ca.jacobian(f_expr,x_sym)
  m = ca.Function('m', [x_sym,w], [m_expr])(x,w) # Remapping from x_sym to x because x is not purely symbolic

  phi = ca.atan(m)
  theta = phi - np.pi/2
  
  a = x
  b = ca.Function('f', [x_sym,w], [f_expr])(x,w) # Remapping from x_sym to x because x is not purely symbolic
  c = a - p*ca.cos(theta)
  d = b - p*ca.sin(theta)
  
  return ca.vertcat(c,d,theta)

# To compare against ground truth error
def generate_path_ground_truth(model,x,w,p):
  x_sym = ca.MX.sym('x')
  f_expr = model(x_sym,w)

  m_expr = ca.jacobian(f_expr,x_sym)
  m = ca.Function('m', [x_sym], [m_expr])(x) # Remapping from x_sym to x because x is not purely symbolic

  phi = ca.atan(m)
  theta = phi - np.pi/2
  
  a = x
  b = ca.Function('f', [x_sym], [f_expr])(x) # Remapping from x_sym to x because x is not purely symbolic
  c = a - p*ca.cos(theta)
  d = b - p*ca.sin(theta)
  
  return ca.vertcat(c,d,theta)

####################
# Create simulator #
####################
from helper_functions import generate_train_and_test_data, create_simulator
train_data, test_data = generate_train_and_test_data()
# test data is the data used to model the surface for the simulator
simulator = create_simulator(test_data)

#####################
# surface modelling #
#####################
from helper_functions import get_h

# h, ndof, MHE_mu, R = get_h(model_type)

# These additional parameters can be used to set a mean other than 0.1 for the RBFs.
# If model_type is not rbf then they are ignored.
h, ndof, MHE_mu, R = get_h(model_type, custom_b_flag=True, custom_b_data=train_data[22].flatten())

######################
# forward kinematics #
######################

# Get casadi function for T_w_tcp.
# Function arguments are the joint states q.
f_fk = simulator.f_fk()

# Get casadi function for T_w_surface.
# Function arguments are the joint states q, as well as the laser distance measurement z.
f_fk_surface = simulator.f_fk_surface()

######################
# task specification #
######################

#define a task context
tc = TaskContext(ts=ts, N=MPC_horizon_length, name=experiment_name)

#define robot joints
q, dq, ddq = tc.define_joint(
    name="q",
    ndof=4,
    resolution="Acceleration"
)

# Progress along surface in world x-direction 
x, dx, ddx = tc.define_joint(
    name="x",
    ndof=1,
    resolution="Acceleration"
)

# Define basis function weights. Used for modelling surface.
w = tc.define_input(
    name ="w",
    ndof = ndof
)

# Desired normal distance from surface.
l = tc.define_input(
    name ="l",
    ndof = 1
)

# Desired progress speed, in terms of progress variable.
s_dot_desired = tc.define_input(
    name ="s_dot_desired",
    ndof = 1
)

# Get expression for transform from world to TCP frame.
T_w_tcp = f_fk(q)

# Extract x,y position of TCP frame
p_w_tcp = T_w_tcp[:2,2]

# Extract orientation of tcp frame
theta_w_tcp = ca.atan2(T_w_tcp[1,0],T_w_tcp[0,0])

# Create geometric path based on model of surface and task paramters
path = generate_path(h, x, w, l)

# # Create geometric path based on ground truth model of surface and task paramters
# ground_truth_surface_model = create_model_rbf(num_basis)
# ground_truth_path = generate_path_ground_truth(ground_truth_surface_model, x, simulator.w, l)
# ground_truth_task_translation_error = p_w_tcp - ground_truth_path[:2]
# ground_truth_task_orientation_error = theta_w_tcp - ground_truth_path[2]

task_translation_error = p_w_tcp - path[:2]
tc.add_task_order2(
  name="task_translation",
  expr= task_translation_error,
  wn=task_translation_wn,
  zeta=task_translation_zeta,
  weight=task_translation_weight,
  ub=task_translation_tolerance,
  lb=-task_translation_tolerance,
  include_last = False
)

task_orientation_error = theta_w_tcp - path[2]
tc.add_task_order2(
  name="task_orientation",
  expr= task_orientation_error,
  wn=task_orientation_wn,
  zeta=task_orientation_zeta,
  weight=task_orientation_weight,
  ub=task_orientation_tolerance,
  lb=-task_orientation_tolerance,
  include_last = False
)

theta_surface = path[2]+np.pi/2
task_progress_speed_error = dx - s_dot_desired*ca.cos(theta_surface)
tc.add_task_order1(
  name="task_progress_speed",
  expr= task_progress_speed_error,
  K=task_progress_speed_K,
  hard=task_progress_speed_hard,
  weight=task_progress_speed_weight,
)

# regularization
tc.add_task_order1(
  name="task_velocity_regularization",
  expr= dq,
  K=3,
  weight=1e-10,
)

tc.add_task_order0(
  name="task_accelerations_regularization",
  expr= ddq,
  weight=1e-10,
)

tc.add_task_order0(
  name="task_accelerations_regularization2",
  expr= ddx,
  weight=1e-10,
)

#position, velocity, and accleration limits

# add position limits
tc.add_task_order2(
  name="joint_0_limit",
  expr=q[0],
  ub=1,
  lb=0,
  hard=False,
  wn=10,
  zeta=1,
  weight=1,
  include_last = False
)

tc.add_task_order2(
  name="joint_1_limit",
  expr=q[1],
  hard=False,
  wn=10,
  zeta=1,
  weight=1,
  ub=np.pi/2,
  lb=-np.pi/2,
  include_last = False
)

# add velocity limits
tc.add_task_order1(
  name="velocity_limits",
  expr=dq,
  ub=velocity_limit,
  lb=-velocity_limit,
  hard=True,
  K=10,
  include_last = False
)

# add acceleration limits
tc.add_task_order0(
  name="acceleration_limits",
  expr=ddq,
  ub=acceleration_limit,
  lb=-acceleration_limit,
  hard=True
)

tc.define_output(
  name="q",
  expr=q,
  only_first=True
)

tc.define_output(
  name="x_MPC_window",
  expr=x,
  only_first=False
)

tc.define_output(
  name="dq",
  expr=dq,
  only_first=True
)

tc.define_output(
  name="ddq",
  expr=ddq,
  only_first=True
)

tc.define_output(
  name="w",
  expr=w,
  only_first=True
)

tc.define_output(
  name="x",
  expr=x,
  only_first=True
)

tc.define_output(
  name="dx",
  expr=dx,
  only_first=True
)

tc.define_output(
  name="ddx",
  expr=ddx,
  only_first=True
)

tc.define_output(
  name="path",
  expr=path,
  only_first=True
)

tc.define_output(
  name="p_w_tcp",
  expr=p_w_tcp,
  only_first=True
)


tc.define_output(
  name="task_translation_error",
  expr=task_translation_error,
  only_first=True
)

tc.define_output(
  name="task_orientation_error",
  expr=task_orientation_error,
  only_first=True
)

tc.define_output(
  name="task_progress_speed_error",
  expr=task_progress_speed_error,
  only_first=True
)


#######################
# create a controller #
#######################

# controller = Solver(tc, N=MPC_horizon_length, ts = ts, mu = mu)
print("Code generating controller...")
controller = generate_solver(tc, solver= "fatrop")

print('DONE')

############################
# create a simulation loop #
############################

#simulation setup

# Initial values
q = q_init
dq = np.array([0.0, 0.0, 0.0, 0.0])
t0 = np.array([0.0])
x = np.array([0])
dx = np.array([0])
l = np.array([0.07]) # Desired distance
s_dot_desired = np.array([s_dot_desired_target])
x_window = [0]

log_list = []
if model_type == 'poly' or model_type == 'RBF' or model_type == 'PPCA':  
  # Create MHE
  mhe = MHE(window_length = MHE_horizon_length, h = h, R = R, mu = MHE_mu)
  
  # Warm start estimator
  # meas = simulator.generateMeasurement(q)
  # x_meas, z_meas = f_fk_surface(q,meas).full()[:2,2]
  # idx = int(x_meas/0.01)
  # for i in range(idx):
  #   _, _ = mhe.step(x=0.01*i, z=test_data[i])
  
  # For keeping buffer from collapsing
  x_meas_prev = 0
  z_meas_prev = 0
  
#simulation loop
dt = 1/freq
t_vec = np.arange(0, T +dt, dt)
for t_step in tqdm(t_vec):
  
    # Get measurement
    meas = simulator.generateMeasurement(q)
    x_meas, z_meas = f_fk_surface(q,meas).full()[:2,2]
  
    # Update estimated surface parameters
    if model_type == 'ground_truth':
        w_est = simulator.w
    elif model_type == 'local_quadratic':
        w_est = estimate_surface_approx_coefficients(simulator.f_surface, x)
    elif model_type == 'poly' or model_type == 'RBF' or model_type == 'PPCA':
        # calculate distance between consecutive measurements
        dist = np.sqrt( (x_meas - x_meas_prev)**2 + (x_meas - x_meas_prev)**2 )
        if dist > buffer_distance_tolerance:
              w_est, x_window = mhe.step(x_meas, z_meas)
              x_meas_prev = x_meas
              z_meas_prev = z_meas
    else: raise NameError('Unsupported model type')

    #run the controller
    out = controller({
        "t0": t0,
        "q0": q,
        "dq0": dq,
        "x0": x,
        "dx0": dx,
        "w": w_est.flatten(),
        "l": l,
        "s_dot_desired": s_dot_desired,
    })
      
    # x_MPC_window = controller.results['x'].full()[-2]
    for k, v in out.items():
        out[k] = v
    # log_list.append({'t': t_step, 'x_window': x_window,'x_MPC_window':x_MPC_window, **out_control, **controller.stats})
    log_list.append({'t': t_step, 'x_window': x_window, **out})

    #integrate joints
    q = q + dt * dq + 0.5 * dt**2 * out["ddq"]
    dq = dq + dt * out["ddq"]
    t0 = t0 + dt
    
    x = x + dt * dx + 0.5 * dt**2 * out["ddx"]
    dx = dx + dt * out["ddx"]
    
# df_cont = pd.DataFrame(log_list)

# utils.plot_output(df_cont,"task_translation_error")

# file_name = time.strftime(experiment_name+"%Y_%m_%d-%H_%M_%S.pickle")
file_name = time.strftime('experiment_data/'+experiment_name+'.pickle')
with open(file_name, 'wb') as handle:
  log = {
    'experiment_parameters': experiment_paramters,
    'log_list': log_list
  }
  pickle.dump(log, handle, protocol=pickle.HIGHEST_PROTOCOL)




  