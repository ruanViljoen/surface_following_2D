from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import numpy as np
from simulator import Simulator
import casadi as ca

def create_dataset(Y_train):
  X_train = np.array(range(11)).reshape(-1,1)
  
  y_vec = np.empty((0,100))

  for y_train in Y_train:
    kernel = 1 * RBF(length_scale=0.2, length_scale_bounds=(1e-2, 1e2))
    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.008)
    gpr.fit(X_train, y_train)
    x_sample = np.linspace(0,10,100).reshape(-1,1)
    
    y_vec = np.concatenate((y_vec,gpr.sample_y(x_sample,10).T/10+0.1), axis=0)
    # plt.plot(x_sample/10, gpr.sample_y(x_sample,10)/10)
    # plt.ylim(-0.1,1)

  return y_vec

def generate_train_and_test_data():
  Y_train = np.array([
      [ 1, 1.2, 1.7, 2 , 2 , 2  , 2 , 1.5, 2  , 1, 1 ],
      [ 1, 0.5, 0.5, 1 , 1 , 0.5, 1 , 1.2, 0.8, 1, 1 ],
      [ 1, 0.6, 0.4, 1 , 1 , 1.5, 1.8 , 2, 1.8, 1, 1 ],
  ])
  
  y_vec = create_dataset(Y_train)
  
  train_data = y_vec[np.array(range(30))!=10]
  train_data = train_data.reshape((train_data.shape[0], train_data.shape[1],1))
  train_data = [trajectory.T for trajectory in train_data]
  test_data = y_vec[21]

  return train_data, test_data

def create_simulator(test_data):
  simulator = Simulator()
  num_basis = 10
  simulator.fit_rbf_linear(test_data,num_basis) # Use custom surface
  # simulator.fit_quadratic(y_vec[11]) # Use custom surface
  # w_actual = simulator.w
  return simulator

def create_model_polynomial(nw):
  x = ca.SX.sym('x')
  w = ca.SX.sym('w',nw)
  model_expr = 0
  
  for i in range(nw):
    model_expr += w[i]*x**i
  
  # model_expr = w[0] + w[1]*x + w[2]*x**2
  return ca.Function('surface_model', [x,w], [model_expr])

def create_model_rbf(num_basis=10, custom_b_flag=False, custom_b_data=None):
  
  # Centre points of rbfs
  C = np.linspace(-0.5, 1.5, num_basis)

  # Parameters of the model
  w = ca.SX.sym('w', num_basis)  # weights
  alpha = 37.4  # shape parameter

  # The model
  x = ca.SX.sym('x')  
  phi = ca.vertcat(*[ca.exp(-alpha * (x - c) ** 2) for c in C])

  if custom_b_flag:
    b_function = fit_rbf(custom_b_data)
    b = b_function(x)
  else:
    b = 0.1
  
  model_expr = phi.T @ w + b
  
  return ca.Function('h', [x,w], [model_expr])

def get_h(model_type, custom_b_flag=False, custom_b_data=None):    
    if model_type == 'ground_truth':
      nw = 10
      h = create_model_rbf(nw)
      MHE_mu, R = None, None
    elif model_type == 'local_quadratic':
      nw = 3
      h = create_model_polynomial(nw)
      
    elif model_type == 'PPCA':
      nw = 10
      model = ProbabilisticPCA(train_data,modes=nw)
      model.train(approx_type='rbf')
      R = np.array([[model.meas_noise**2]])
      
      # Create h function
      x = ca.SX.sym('x')
      w = ca.SX.sym('w',nw)
      H = model.H_ca_function(x)
      b = model.b_ca_function(x)
      h = ca.Function('h', [x,w], [H@w + b])
      MHE_mu = 0.5
      
    elif model_type == 'RBF':
      nw = 20
      R = np.array([[0.001]])
      h = create_model_rbf(nw,custom_b_flag, custom_b_data)
      MHE_mu = 1e-4
      
    elif model_type == 'poly':
      nw = 3
      R = np.array([[0.001]])
      h = create_model_polynomial(nw)
      MHE_mu = 1e-5
      
    else: raise NameError('Unsupported model type') 
      
    surface_model = h
    ndof = nw

    return h, ndof, MHE_mu, R

def fit_rbf(Y_train,num_basis=10):

  X_train = np.linspace(0,1,100)
  
  # Centers (chosen uniformly)
  C = np.linspace(-0.5, 1.5, num_basis)
  b = ca.SX.sym('b')  # bias

  # Parameters of the model
  w = ca.SX.sym('w', num_basis)  # weights
  alpha = ca.SX.sym('alpha')  # shape parameter

  # The model
  x = ca.SX.sym('x')
  phi = ca.vertcat(*[ca.exp(-alpha * (x - c) ** 2) for c in C])
  
  y = phi.T @ w + b
  f = ca.Function('f', [x, w, b, alpha], [y])

  # Define the error function
  error = 0
  for i in range(len(X_train)):
    error += (Y_train[i] - f(X_train[i], w, b, alpha)) ** 2
  # Choose an optimizer
  opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.tol": 1e-5}
  decision_vars = ca.vertcat(w, b, alpha)
  g=b
  nlp = {'x': ca.vertcat(w, b, alpha), 'f': error, 'g':g}
  # nlp = {'x': ca.vertcat(w, b, alpha), 'f': error}
  solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
  
  b_desired = 0.1

  # Initial guess
  w0 = np.ones((num_basis))
  # b0 = 1
  b0 = b_desired
  alpha0 = np.array([1])
  x0 = ca.vertcat(w0,b0,alpha0)
  # Solve
  res = solver(x0=x0,ubg=b_desired,lbg=b_desired)
  # res = solver(x0=x0)

  # Extract optimal values
  w_opt, b_opt, alpha_opt = np.split(res['x'].full().flatten(), [num_basis, num_basis + 1])

  return ca.Function('f_surface', [x], [f(x, w_opt, b_opt, alpha_opt)])