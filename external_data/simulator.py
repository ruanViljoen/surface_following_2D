import numpy as np
import casadi as ca
import plotly.graph_objects as go
import matplotlib.pyplot as plt

class Shape:
  def __init__(self, points, pathSequence):
    self.P = points
    self.pathSequence = pathSequence

  def getPath(self,T):
    P_shape = self.P.shape
    P = np.ones((P_shape[0]+1, P_shape[1]))
    P[:P_shape[0]] = self.P

    P = T @ P

    pathSequence = self.pathSequence


    x_path = []
    y_path = []

    for i in pathSequence:
      x_path +=[P[0,i-1]]
      y_path +=[P[1,i-1]]

    return x_path, y_path

def createShape_link(length):
  width = 0.0125

  P = []
  P += [[0, -width/2]] # p1
  P += [[length, -width/2]] # p2
  P += [[length, width/2]] # p3
  P += [[0, width/2]] # p4

  P = np.array(P).T
  pathSequence = [1,2,3,4,1]

  shape = Shape(
      points = P,
      pathSequence = pathSequence
  )

  return shape

def createShape_ee():

  lx1 = 0.005
  lx2 = 0.01
  lx3 = 0.01
  lx4 = 0.02
  
  ly1 = 0.01
  ly2 = 0.01
  ly3 = 0.025
  ly4 = 0.005
  ly5 = 0.005

  x = 0
  y = - ( ly1 + ly2 + ly3 + ly4/2 )

  P = []
  P += [[x - lx1 - lx2, y + ly1 + ly2 + ly3 + ly4 + ly5]] # p1
  P += [[x + lx1 + lx2, y + ly1 + ly2 + ly3 + ly4 + ly5]] # p2
  P += [[x + lx1 + lx2, y + ly1 + ly2 + ly3 + ly4]] # p3
  P += [[x + lx1 + lx2 + lx3, y + ly1 + ly2 + ly3 + ly4]] # p4
  P += [[x + lx1 + lx2 + lx3, y + ly1 + ly2 + ly3 + ly4 + ly5]] # p5
  P += [[x + lx1 + lx2 + lx3 + lx4, y + ly1 + ly2 + ly3 + ly4 + ly5]] # p6
  P += [[x + lx1 + lx2 + lx3 + lx4, y + ly1 + ly2 + ly3 - ly5]] # p7
  P += [[x + lx1 + lx2 + lx3, y + ly1 + ly2 + ly3 - ly5]] # p8
  P += [[x + lx1 + lx2 + lx3, y + ly1 + ly2 + ly3]] # p9
  P += [[x + lx1 + lx2, y + ly1 + ly2 + ly3]] # p10
  P += [[x + lx1 + lx2, y + ly1 + ly2]] # p11
  P += [[x + lx1, y + ly1 + ly2]] # p12
  P += [[x + lx1, y + ly1]] # p13
  P += [[x, y]] # p14
  P += [[x - lx1, y + ly1]] # p15
  P += [[x - lx1, y + ly1 + ly2]] # p16
  P += [[x - lx1 - lx2, y + ly1 + ly2]] # p17

  P = np.array(P).T

  P_shape = P.shape
  ones = np.ones((P_shape[0]+1, P_shape[1]))
  ones[:P_shape[0]] = P

  
  T = np.array([
    [np.cos(np.pi/2), -np.sin(np.pi/2), 0],
    [np.sin(np.pi/2), np.cos(np.pi/2), 0],
    [0, 0, 1]
  ])  

  P = (T @ ones)[:P_shape[0]]

  pathSequence = [1,2,3,4,5,6,7,8,4,9,10,3,11,12,13,14,15,16,12,17,1]

  shape = Shape(
      points = P,
      pathSequence = pathSequence
  )

  return shape

def createTransformationMatrix(x_delta, y_delta,theta_delta):
  return np.array([
    [np.cos(theta_delta), -np.sin(theta_delta), x_delta],
    [np.sin(theta_delta), np.cos(theta_delta), y_delta],
    [0, 0, 1]
  ])

def createTransformationMatrixCasadi(x_delta, y_delta,theta_delta):
  return ca.vertcat(
    ca.cos(theta_delta), -ca.sin(theta_delta), x_delta,
    ca.sin(theta_delta), ca.cos(theta_delta), y_delta,
    0,                   0,                   1
  ).reshape((3,3)).T

class Simulator:
  def __init__(self):

    # Create robot

    # Parameters
    self.link_length = 0.15

    self.shape_link0 = createShape_link(
        length = 1
    )

    self.shape_link1 = createShape_link(
        length = self.link_length
    )
    self.shape_link2 = createShape_link(
        length = self.link_length
    )
    self.shape_ee = createShape_ee()
    
    self.create_f_surface()
    
    self.x0 = 1 # Initial guess for get_measurement function

#     self.q = ca.vertcat(q_init)

#   def step(self, q_dot, dt):
#     self.q += ca.vertcat(q_dot)*dt # Euler integration
#     z = self.generateMeasurement(self.q)
#     return self.q.full().flatten(), z

  def f_fk(self):
    
    q0 = ca.SX.sym('q0')
    q1 = ca.SX.sym('q1')
    q2 = ca.SX.sym('q2')
    q3 = ca.SX.sym('q3')
        
    T_w_link0 = createTransformationMatrixCasadi(0,0.5,0)
    T_link0_link1 = createTransformationMatrixCasadi(q0,0,q1-np.pi/2)
    T_link1_link2 = createTransformationMatrixCasadi(self.link_length,0,q2)
    T_link2_ee = createTransformationMatrixCasadi(self.link_length,0,q3)
    T_ee_tcp = createTransformationMatrixCasadi(0.05,0,0)
    
    T_w_link1 = T_w_link0 @ T_link0_link1
    T_w_link2 = T_w_link1 @ T_link1_link2
    T_w_ee = T_w_link2 @ T_link2_ee
    T_w_tcp = T_w_ee @ T_ee_tcp
    
    return ca.Function('fk', [ca.vertcat(q0,q1,q2,q3)], [T_w_tcp])
  
  def f_fk_surface(self):
    
    q0 = ca.SX.sym('q0')
    q1 = ca.SX.sym('q1')
    q2 = ca.SX.sym('q2')
    q3 = ca.SX.sym('q3')
    z = ca.SX.sym('z')
        
    T_w_link0 = createTransformationMatrixCasadi(0,0.5,0)
    T_link0_link1 = createTransformationMatrixCasadi(q0,0,q1-np.pi/2)
    T_link1_link2 = createTransformationMatrixCasadi(self.link_length,0,q2)
    T_link2_ee = createTransformationMatrixCasadi(self.link_length,0,q3)
    T_ee_sensor = createTransformationMatrixCasadi(0.0075, 0.035,0)
    T_sensor_surface = createTransformationMatrixCasadi(z,0,0)
    
    T_w_link1 = T_w_link0 @ T_link0_link1
    T_w_link2 = T_w_link1 @ T_link1_link2
    T_w_ee = T_w_link2 @ T_link2_ee
    T_w_sensor = T_w_ee @ T_ee_sensor
    T_w_surface = T_w_sensor @ T_sensor_surface
    
    return ca.Function('fk', [ca.vertcat(q0,q1,q2,q3),z], [T_w_surface])
  
  def fit_quadratic(self,data):
    x = np.linspace(0,1,100)
    y = data

    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression

    #specify degree of 3 for polynomial regression model
    #include bias=False means don't force y-intercept to equal zero
    degree = 10
    poly = PolynomialFeatures(degree=degree, include_bias=False)

    #reshape data to work properly with sklearn
    poly_features = poly.fit_transform(x.reshape(-1, 1))

    #fit polynomial regression model
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(poly_features, y)

    #display model coefficients

    w = poly_reg_model.coef_
    x_sym = ca.SX.sym('x')

    f_expr = poly_reg_model.intercept_
    for i in range(degree):
      f_expr += w[i]*x_sym**(i+1)

    self.f_surface = ca.Function('f_surface', [x_sym], [f_expr])

    x_test = np.linspace(-1,2,100)
    y_fit = self.f_surface(x_test).full().flatten()
    
    plt.plot(x_test,y_fit)  
    plt.plot(x,y)
    plt.ylim(-0.1,1)
    plt.legend(['fit','data'])
    
    surface_x = np.linspace(-0.1,1.1,1200)
    surface_y = np.array([ self.f_surface(surface_x[i]).full().flatten()[0] for i in range(len(surface_x)) ])
    self.surface = np.array([
      surface_x,
      surface_y
    ]) 
    
  def fit_rbf(self,Y_train,num_basis=10):

    X_train = np.linspace(0,1,100)
    
    # Centers (chosen uniformly)
    C = np.linspace(0, 1, num_basis)

    # Parameters of the model
    w = ca.SX.sym('w', num_basis)  # weights
    b = ca.SX.sym('b')  # bias
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

    self.f_surface = ca.Function('f_surface', [x], [f(x, w_opt, b_opt, alpha_opt)])

    x_test = np.linspace(-1,2,100)
    y_fit = self.f_surface(x_test).full().flatten()
    
    plt.plot(x_test,y_fit)  
    plt.plot(X_train,Y_train)
    plt.ylim(-0.1,1)
    plt.legend(['fit','data'])
    
    surface_x = np.linspace(-0.1,1.1,1200)
    surface_y = np.array([ self.f_surface(surface_x[i]).full().flatten()[0] for i in range(len(surface_x)) ])
    self.surface = np.array([
      surface_x,
      surface_y
    ]) 
    self.w = np.concatenate((w_opt, b_opt, alpha_opt))
    
  def fit_rbf_linear_old(self,Y_train,num_basis=10):

    X_train = np.linspace(0,1,100)
    
    # Centers (chosen uniformly)
    C = np.linspace(0, 1, num_basis)

    # Parameters of the model
    w = ca.SX.sym('w', num_basis)  # weights
    b = 0.1
    alpha = 37.4

    # The model
    x = ca.SX.sym('x')
    phi = ca.vertcat(*[ca.exp(-alpha * (x - c) ** 2) for c in C])
    y = phi.T @ w + b
    f = ca.Function('f', [x, w], [y])

    # Define the error function
    error = 0
    for i in range(len(X_train)):
      error += (Y_train[i] - f(X_train[i], w)) ** 2
    # Choose an optimizer
    opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.tol": 1e-5}
    nlp = {'x': w, 'f': error}
    # nlp = {'x': ca.vertcat(w, b, alpha), 'f': error}
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
    
    # Initial guess
    w0 = np.ones((num_basis))
    # Solve
    res = solver(x0=w0)

    # Extract optimal values
    w_opt = res['x'].full().flatten()

    self.f_surface = ca.Function('f_surface', [x], [f(x, w_opt)])

    x_test = np.linspace(-1,2,100)
    y_fit = self.f_surface(x_test).full().flatten()
    
    plt.plot(x_test,y_fit)  
    plt.plot(X_train,Y_train)
    plt.ylim(-0.1,1)
    plt.legend(['fit','data'])
    
    surface_x = np.linspace(-0.1,1.1,1200)
    surface_y = np.array([ self.f_surface(surface_x[i]).full().flatten()[0] for i in range(len(surface_x)) ])
    self.surface = np.array([
      surface_x,
      surface_y
    ]) 
    self.w = w_opt
  
  def fit_rbf_linear(self,Y_train,num_basis=10):
    
    N = 100
    b = 0.1
    alpha = 37.4
    C = np.linspace(0, 1, num_basis)
    
    x_vec = np.linspace(0,1,N)
    y_vec = Y_train
    
    c = (y_vec-b).reshape(-1,1)
    
    A = np.empty((0,num_basis))
    for x in x_vec:
      phi = np.array([np.exp(-alpha * (x - centre) ** 2) for centre in C]).reshape(1,-1)
      A = np.concatenate((A,phi),axis=0)
      
    w = np.linalg.pinv(A)@c
    self.w = w
    
    x = ca.SX.sym('x')
    phi = ca.vertcat(*[ca.exp(-alpha * (x - c) ** 2) for c in C])
    y = phi.T @ w + b
    self.f_surface = ca.Function('f_surface', [x], [y])

    x_test = np.linspace(-1,2,100)
    y_fit = self.f_surface(x_test).full().flatten()
    
    plt.plot(x_test,y_fit)  
    plt.plot(x_vec,Y_train)
    plt.ylim(-0.1,1)
    plt.legend(['fit','data'])
    
    surface_x = np.linspace(-0.1,1.1,1200)
    surface_y = np.array([ self.f_surface(surface_x[i]).full().flatten()[0] for i in range(len(surface_x)) ])
    self.surface = np.array([
      surface_x,
      surface_y
    ]) 
    return
    
  def create_f_surface(self, from_data=False, data=None):
    # for f_temp in f:
    #   x_min, x_max = f_temp['domain']
    #   if (x >= x_min) and (x <= x_max):
    #     return f_temp['function'](x)

#     y = ca.if_else(
#       ca.logic_and(x>0.2, x<0.4), # Condition
#       0.5*x - 0.1, # If true,
#       ca.if_else(
#         ca.logic_and(x>=0.4, x <0.8),
#         2.5*x**2 - 3*x +0.9,
#         ca.if_else(
#           ca.logic_and(x>=0.8, x <1),
#           -0.5*x + 0.5,
#           0
#         )
#       )
#     ) 

#     y = ca.if_else(
#       ca.logic_and(x>0.3, x<0.7), # Condition
#       0.5*x - 0.15, # If true
#       ca.if_else(
#         x>=0.7,
# #         ca.logic_and(, x <0.8),
#         0.2,
#         0
#       )
#     ) 

    
    if from_data:
      self.f_surface = self.fit_quadratic(data)

  
    x = ca.SX.sym('x')
    surface_expr = ca.vertcat(0.2 + 0.09*( 1/(1 + ca.exp( -30*( x-0.4 ) )) - 1/(1 + ca.exp( -30*( x-0.7 ) )) ))
    self.f_surface = ca.Function('f_surface', [x], [surface_expr])
    
    
    surface_x = np.linspace(-0.1,1.1,1200)
    surface_y = np.array([ self.f_surface(surface_x[i]).full().flatten()[0] for i in range(len(surface_x)) ])
    self.surface = np.array([
      surface_x,
      surface_y
    ])
    # y = self.surface_expr

    # if symbolic:
    #   return y
    # else:
    #   return y.full().flatten()[0]

  def generateAnimation(self, q_vec, z_vec, opacity_vec, extra_frame_vec, extra_surface_vec, show_tf=False, slow=False):
    N = len(q_vec)
    fig = go.Figure(
        data= self.generateFrameWithExtra(q_vec[0], z_vec[0], [extra_frame_vec[0]], opacity_vec, extra_surface_vec[0], show_tf=show_tf, stationary=True, surface=self.surface) if not slow else self.generateFrameSlow(q_vec[:,0], z_vec[0], show_tf=show_tf, surface=self.surface),
        frames=[
          go.Frame(    
              data=self.generateFrameWithExtra(q_vec[k], z_vec[k], [extra_frame_vec[k]], opacity_vec, extra_surface_vec[k], show_tf=show_tf) if not slow else self.generateFrameSlow(q_vec[:,k], z_vec[k], show_tf=show_tf, surface=self.surface)
          ) for k in range(N)
        ],
        layout=go.Layout(
            showlegend=False,
            # autosize=True,
            width=1200,
            height=750,
            xaxis=dict(range=[-0.05, 1.05], autorange=False, zeroline=False),
            yaxis=dict(scaleanchor="x", scaleratio=1),
            # title_text="Contour following",
            paper_bgcolor='rgba(0.99,0.99,0.99,0)',
            plot_bgcolor='rgba(0.95,0.95,0.95,1)',
            updatemenus = [
                {
                    "buttons": [
                        {
                            "args": [
                              None,
                              {
                                "frame": {"duration": 5, "redraw":False},
                                "mode": "immediate",
                                "fromcurrent": True,
                                "transition": {"duration": 5, "easing": "linear"},
                              }  
                              ], #,
                            "label": "&#9654;", # play symbol
                            "method": "animate",
                        },
                        # {
                        #     "args": [None, frame_args(0)],
                        #     "label": "&#9724;", # pause symbol
                        #     "method": "animate",
                        # },
                    ],
                    "direction": "right",
                    "pad": {"r": 100, "t": 70},
                    "type": "buttons",
                    "x": 0.5,
                    "y": 0,
                }
            ],
        ),
        layout_yaxis_range=[-0.001,0.6],
    )
    return fig
  
  def generateRobotPath(self,q,z):
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
        
    T_w_link0 = createTransformationMatrix(0,0.5,0)
    T_link0_link1 = createTransformationMatrix(q0,0,q1-np.pi/2)
    T_link1_link2 = createTransformationMatrix(self.link_length,0,q2)
    T_link2_ee = createTransformationMatrix(self.link_length,0,q3)
    T_ee_sensor = createTransformationMatrix(0.0075, 0.035,0)
    T_sensor_surface = createTransformationMatrix(z,0,0)

    T_w_link1 = T_w_link0 @ T_link0_link1
    T_w_link2 = T_w_link1 @ T_link1_link2
    T_w_ee = T_w_link2 @ T_link2_ee
    T_w_sensor = T_w_ee @ T_ee_sensor
    T_w_surface = T_w_sensor @ T_sensor_surface

    tfs = [
      {
        'name': 'link1',
        'tf': T_w_link1
      },
      {
        'name': 'link2',
        'tf': T_w_link2
      },
      {
        'name': 'ee',
        'tf': T_w_ee
      },
      {
        'name': 'surface',
        'tf': T_w_surface
      },
    ]
    
    path_link0 = self.shape_link0.getPath(T_w_link0)
    path_link1 = self.shape_link1.getPath(T_w_link1)
    path_link2 = self.shape_link2.getPath(T_w_link2)
    path_ee = self.shape_ee.getPath(T_w_ee)
    
    path = [
      path_link0[0] + [None] + path_link1[0] + [None] + path_ee[0] + [None] + path_link2[0], # + [None] + path_ee[0], # x
      path_link0[1] + [None] + path_link1[1] + [None] + path_ee[1]+ [None] + path_link2[1]  # y
    ]
    
    joints = [
        [
        T_w_link1[0,2],
        T_w_link2[0,2],
        T_w_ee[0,2],
      ],
      [
        T_w_link1[1,2],
        T_w_link2[1,2],
        T_w_ee[1,2],
      ]
    ]
    
    laser = [
      [
        T_w_sensor[0,2],
        T_w_surface[0,2]
      ],
      [
        T_w_sensor[1,2],
        T_w_surface[1,2]
      ]
    ]
    
#     frames_x = [
#       [0,0.05], # x
#       [0,0]  # y
#     ]
    tf_x = np.array([
      [0, 0.05], # x
      [0, 0],    # y
      [1, 1]
    ])

    tf_y = np.array([
      [0, 0], # x
      [0, 0.05],    # y
      [1, 1]
    ])
    
    frames_x = [
      tf_x[0].tolist(), # x
      tf_x[1].tolist(), # y
    ]

    frames_y= [
      tf_y[0].tolist(), # x
      tf_y[1].tolist(), # y
    ]    
    
    for tf in tfs:
      temp_tf_x = tf['tf'] @ tf_x
      temp_tf_y = tf['tf'] @ tf_y
      
      frames_x[0] += [None] + temp_tf_x[0].tolist()
      frames_x[1] += [None] + temp_tf_x[1].tolist()
    
      frames_y[0] += [None] + temp_tf_y[0].tolist()
      frames_y[1] += [None] + temp_tf_y[1].tolist()

#       test1_x = T_w_ee @ test_x
#     test1_y = T_w_ee @ test_y
    

    
#     frames_x = [
#       test_x[0].tolist() + [None] + test1_x[0].tolist(), # x
#       test_x[1].tolist() + [None] + test1_x[1].tolist(), # y
#     ]

#     frames_y= [
#       test_y[0].tolist() + [None] + test1_y[0].tolist(), # x
#       test_y[1].tolist() + [None] + test1_y[1].tolist(), # y
#     ]

    
    return path, joints, laser, frames_x, frames_y   
      
  def generateFrame(self, q, z, show_tf=True, stationary=False, surface=None):
    
    path, joints, laser, frames_x, frames_y = self.generateRobotPath(q,z)
    
    data = [
      go.Scatter( # Robot1 path
        x=path[0], y=path[1],
        mode="lines",
        line=dict(width=2, color="black"),
        fill='toself', 
        fillcolor = 'rgb(143, 161, 204)',
        name='robot_frame'
      ),
      go.Scatter( # Laser distance measurement
        x=laser[0], y=laser[1],
        mode="lines+markers",
        line=dict(width=2, color="red"),
        marker=dict(color="red", size=7)
      ),
      go.Scatter( # Joints
        x=joints[0],
        y=joints[1],
        mode="markers",
        marker=dict(color="grey", size=15, line=dict(
            color='black',
            width=2
        ))
      ),
    ]
    
    if show_tf:
      data += [
        go.Scatter( # Frames x
          x=frames_x[0], y=frames_x[1],
          mode="lines",
          line=dict(width=5, color="rgb(255,0,0)"),
        ),
        go.Scatter( # Frames y
          x=frames_y[0], y=frames_y[1],
          mode="lines",
          line=dict(width=5, color="rgb(0, 255, 0)"),
        ),
      ]

    if stationary:
      data += [
        go.Scatter( # Surface
          x=surface[0], y=surface[1],
          mode="lines",
          line=dict(width=2, color="black"),
          fill='tozeroy', 
          fillcolor = 'grey',
        ),
      ]

    return data
  
  def generateFrameSlow(self, q, z, surface, show_tf=True):
    
    path, joints, laser, frames_x, frames_y = self.generateRobotPath(q,z)
    
    data = [
      go.Scatter( # Surface
        x=surface[0], y=surface[1],
        mode="lines",
        line=dict(width=2, color="black"),
        fill='tozeroy', 
        fillcolor = 'grey',
      ),
      go.Scatter( # Robot1 path
        x=path[0], y=path[1],
        mode="lines",
        line=dict(width=2, color="black"),
        fill='toself', 
        fillcolor = 'rgb(143, 161, 204)',
        name='robot_frame'
      ),
      go.Scatter( # Laser distance measurement
        x=laser[0], y=laser[1],
        mode="lines+markers",
        line=dict(width=2, color="red"),
        marker=dict(color="red", size=7)
      ),
      go.Scatter( # Joints
        x=joints[0],
        y=joints[1],
        mode="markers",
        marker=dict(color="grey", size=15, line=dict(
            color='black',
            width=2
        ))
      ),
    ]
    
    if show_tf:
      data +=[
        go.Scatter( # Frames x
          x=frames_x[0], y=frames_x[1],
          mode="lines",
          line=dict(width=5, color="rgb(255,0,0)"),
        ),
        go.Scatter( # Frames y
          x=frames_y[0], y=frames_y[1],
          mode="lines",
          line=dict(width=5, color="rgb(0, 255, 0)"),
        ),        
      ]
    
    return data

  def generateMeasurement(self,q):
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    T_w_link0 = createTransformationMatrix(0,0.5,0)
    T_link0_link1 = createTransformationMatrix(q0,0,q1-np.pi/2)
    T_link1_link2 = createTransformationMatrix(self.link_length,0,q2)
    T_link2_ee = createTransformationMatrix(self.link_length,0,q3)
    T_ee_sensor = createTransformationMatrix(0.0075, 0.035,0)

    T_w_link1 = T_w_link0 @ T_link0_link1
    T_w_link2 = T_w_link1 @ T_link1_link2
    T_w_ee = T_w_link2 @ T_link2_ee
    T_w_sensor = T_w_ee @ T_ee_sensor

    x_sensor = T_w_sensor[0,2]
    y_sensor = T_w_sensor[1,2]
    theta_sensor = np.arctan2(T_w_sensor[1,0], T_w_sensor[0,0])

    x_s = ca.SX.sym('x_s')
    y_s = self.f_surface(x_s)

    r = ca.SX.sym('r')
    w = ca.vertcat(x_s,r) # Decision variables

    eps_x = x_sensor + np.cos(theta_sensor)*r - x_s
    eps_y = y_sensor + np.sin(theta_sensor)*r - y_s

    eps = eps_x**2 + eps_y**2

    g = []
#     g = ca.vertcat(x_s)
    f_cost = eps
    
    nlp = {'x':w, 'f':f_cost, 'g':g}
    
    
    
    kkt_tol_pr = 1e-6
    kkt_tol_du = 1e-6
    min_step_size = 1e-10
    max_iter = 10
    max_iter_ls = 3
    qpsol_options = {
        "constr_viol_tol": kkt_tol_pr,
        "dual_inf_tol": kkt_tol_du,
        # "min_step_size": min_step_size,
        "verbose": False,
        "print_iter": False,
        "print_header": False,
        "dump_in": False,
        "error_on_fail": False,
    }
    solver_options = {
        "qpsol": "qrqp",
        # "qpsol": "ocqp",
        # "hessian_approximation" :"limited-memory",
        "qpsol_options": qpsol_options,
        "verbose": False,
        "tol_pr": kkt_tol_pr,
        "tol_du": kkt_tol_du,
        "min_step_size": min_step_size,
        # "max_iter": max_iter,
        # "max_iter_ls": max_iter_ls,
        "print_iteration": False,
        "print_header": False,
        "print_status": False,
        "print_time": False,
        # "convexify_strategy":"regularize"
        "error_on_fail": False,
    }  # "convexify_strategy":"regularize"
        # tc.set_ocp_solver("sqpmethod", solver_options)
       
    S = ca.nlpsol('S', 'sqpmethod', nlp, solver_options)
    
#     S = ca.nlpsol('S', 'ipopt', nlp, {'ipopt':{'print_level':0}, 'print_time': False})
    # r = S(x0=[1]*2,lbg=[],ubg=[])
    r = S(x0=self.x0,lbg=[],ubg=[])

    w_opt = r['x'].full().flatten()
    x_s_opt = w_opt[0]
    y_s_opt = self.f_surface(x_s_opt)
    r_opt = w_opt[1]
    self.x0 = r['x']

    eps_x_opt = x_sensor + np.cos(theta_sensor)*r_opt - x_s_opt
    eps_y_opt = y_sensor + np.sin(theta_sensor)*r_opt - y_s_opt
    eps_opt = eps_x_opt**2 + eps_y_opt**2
    z = r_opt
    if (x_s_opt <= 0.0) or (x_s_opt >= 1.0) or (r_opt < 0):
#     if (eps_opt > 1e-10) or (r_opt < 0):
      z = 1e-3

    # return x_s_opt, y_s_opt.full().flatten()[0]
    return z 

  def generateFig(self,q):

    z = self.generateMeasurement(q)
    fig = go.Figure(
      data = self.generateFrame(q, z, True, True, self.surface),
      layout=go.Layout(
        showlegend=False,
        # autosize=True,
        width=1200,
        height=750,
        xaxis=dict(range=[-0.05, 1.05], autorange=False, zeroline=False),
        yaxis=dict(scaleanchor="x", scaleratio=1),
        # ticks='',
        paper_bgcolor='rgba(0.99,0.99,0.99,0)',
        plot_bgcolor='rgba(0.95,0.95,0.95,1)',
        # title_text="Contour following",
      ),
      layout_yaxis_range=[-0.001,0.6]
    )
    
#   def generate_quadratic(self, x):
    
#     # Get surface expression
#     x_sym = ca.SX.sym('x')
#     f = self.f_surface(x_sym, symbolic=True)
    
#     # Find quadratic approximation at x
#     df = ca.Function('jacobian', [x_sym],[ca.jacobian(f,x_sym)])(x).full().flatten()[0]
#     ddf = ca.Function('hessian', [x_sym],[ca.hessian(f,x_sym)[0]])(x).full().flatten()[0]
    
#     f_approx = ca.Function('f_approx', [x_sym], [self.f_surface(x) + df*(x_sym - x) + 0.5*ddf*(x_sym - x)**2])
    
#     x_range = list(np.linspace(x-0.2,x+0.2,20))
#     y_range = list(f_approx(x_range).full().flatten())
    
#     # Find values of quadratic for x_range = [x-0.1, x+0.1]
    
#     # return
#     return [x_range,y_range]

  def generate_frame(self, frame):
    x = frame[0]
    y = frame[1]
    theta = frame[2]
    
    tf_x = np.array([
      [0, 0.05], # x
      [0, 0],    # y
      [1, 1]
    ])

    tf_y = np.array([
      [0, 0],    # x
      [0, 0.05], # y
      [1, 1]
    ])

    tf = createTransformationMatrix(x,y,theta)

    temp_tf_x = tf @ tf_x
    temp_tf_y = tf @ tf_y

    frames_x = [
      temp_tf_x[0].tolist(), # x
      temp_tf_x[1].tolist(), # y
    ]

    frames_y= [
      temp_tf_y[0].tolist(), # x
      temp_tf_y[1].tolist(), # y
    ]

    return frames_x, frames_y
    
  def generateFrameWithExtra(self, q_vec, z_vec, extra_frames, opacity_vec, extra_surfaces, show_tf=True, stationary=False, surface=None):
    
    q = q_vec[0]
    z = z_vec[0]
    # extra_frame = extra_frames[0]
    
    path, joints, laser, frames_x, frames_y = self.generateRobotPath(q,z)
    # quadratic = self.generate_quadratic(x)
    
    data = []
    # Generate robots
    for i in range(len(q_vec)):
      q = q_vec[i]
      z = z_vec[i]
      opacity = opacity_vec[i]
      path, joints, laser, frames_x, frames_y = self.generateRobotPath(q,z)
      data += [
        go.Scatter( # Robot1 path
          x=path[0], y=path[1],
          mode="lines",
          line=dict(width=2, color="black"),
          fill='toself', 
          fillcolor = 'rgb(143, 161, 204)',
          name='robot_frame',
          opacity = opacity
        ),
        go.Scatter( # Joints
          x=joints[0],
          y=joints[1],
          mode="markers",
          marker=dict(color="grey", size=15, line=dict(
              color='black',
              width=2
          )),
          opacity = opacity
          
        ),
      ]
      if z != 0:
        data += [
          go.Scatter( # Laser distance measurement
            x=laser[0], y=laser[1],
            mode="lines+markers",
            line=dict(width=2, color="red"),
            marker=dict(color="red", size=7)
          ),
        ]
      
    for extra_frame in extra_frames:
      extra_frames_x, extra_frames_y = self.generate_frame(extra_frame)
      data += [
        go.Scatter( # Frames x
          x=extra_frames_x[0], y=extra_frames_x[1],
          mode="lines",
          line=dict(width=5, color="rgb(255,0,0)"),
        ),
        go.Scatter( # Frames y
          x=extra_frames_y[0], y=extra_frames_y[1],
          mode="lines",
          line=dict(width=5, color="rgb(0, 255, 0)"),
        ),
      ]
      
    for i in range(len(extra_surfaces)):

      if i == 0:
        data += [
          go.Scatter( # extra
            x=extra_surfaces[i][0],
            y=extra_surfaces[i][1],
            mode="lines",
            opacity=1,
            line=dict(width=6, color="blue"),
          ),
        ]
      elif i == 1:
        data += [
          go.Scatter( # extra
            x=extra_surfaces[i][0],
            y=extra_surfaces[i][1]-0.01,
            mode="lines",
            opacity=1,
            line=dict(width=6, color="red"),
          ),
        ]
      elif i == 2: 
        data += [
          go.Scatter( # extra
            x=extra_surfaces[i][0],
            y=extra_surfaces[i][1],
            mode="lines",
            opacity=0.5,
            line=dict(width=2, color="green"),
          ),
        ]
    if show_tf:
      data += [
        go.Scatter( # Frames x
          x=frames_x[0], y=frames_x[1],
          mode="lines",
          line=dict(width=5, color="rgb(255,0,0)"),
        ),
        go.Scatter( # Frames y
          x=frames_y[0], y=frames_y[1],
          mode="lines",
          line=dict(width=5, color="rgb(0, 255, 0)"),
        ),
      ]

    if stationary:
      data += [
        go.Scatter( # Surface
          x=surface[0], y=surface[1],
          mode="lines",
          line=dict(width=2, color="black"),
          # fill='tozeroy', 
          # fillcolor = 'white',
        ),
      ]

    return data
  
  def generateFrameWithExtra_old(self, q, z, extra_frame, extra_surface, show_tf=True, stationary=False, surface=None):
    
    # extra_frame = extra_frames[0]
    
    path, joints, laser, frames_x, frames_y = self.generateRobotPath(q,z)
    # quadratic = self.generate_quadratic(x)
    
    data = []
#     data += [
#       go.Scatter( # Robot1 path
#         x=path[0], y=path[1],
#         mode="lines",
#         line=dict(width=2, color="black"),
#         fill='toself', 
#         fillcolor = 'rgb(143, 161, 204)',
#         name='robot_frame',
#       ),
#       go.Scatter( # Joints
#         x=joints[0],
#         y=joints[1],
#         mode="markers",
#         marker=dict(color="grey", size=15, line=dict(
#             color='black',
#             width=2
#         )),
#       ),
#     ]
#     if z != 0:
#       data += [
#         go.Scatter( # Laser distance measurement
#           x=laser[0], y=laser[1],
#           mode="lines+markers",
#           line=dict(width=2, color="red"),
#           marker=dict(color="red", size=7)
#         ),
#       ]
      
#     extra_frames_x, extra_frames_y = self.generate_frame(extra_frame)
#     data += [
#       go.Scatter( # Frames x
#         x=extra_frames_x[0], y=extra_frames_x[1],
#         mode="lines",
#         line=dict(width=5, color="rgb(255,0,0)"),
#       ),
#       go.Scatter( # Frames y
#         x=extra_frames_y[0], y=extra_frames_y[1],
#         mode="lines",
#         line=dict(width=5, color="rgb(0, 255, 0)"),
#       ),
#     ]
      
    data += [
      go.Scatter( # Robot1 path
        x=path[0], y=path[1],
        mode="lines",
        line=dict(width=2, color="black"),
        fill='toself', 
        fillcolor = 'rgb(143, 161, 204)',
        name='robot_frame'
      ),
      go.Scatter( # Laser distance measurement
        x=laser[0], y=laser[1],
        mode="lines+markers",
        line=dict(width=2, color="red"),
        marker=dict(color="red", size=7)
      ),
      go.Scatter( # Joints
        x=joints[0],
        y=joints[1],
        mode="markers",
        marker=dict(color="grey", size=15, line=dict(
            color='black',
            width=2
        ))
      ),
      go.Scatter( # extra
        x=extra_surface[0],
        y=extra_surface[1],
        mode="lines",
        opacity=0.5,
        line=dict(width=2, color="green"),
      ),
    ]
    
    if show_tf:
      data += [
        go.Scatter( # Frames x
          x=frames_x[0], y=frames_x[1],
          mode="lines",
          line=dict(width=5, color="rgb(255,0,0)"),
        ),
        go.Scatter( # Frames y
          x=frames_y[0], y=frames_y[1],
          mode="lines",
          line=dict(width=5, color="rgb(0, 255, 0)"),
        ),
      ]

    if stationary:
      data += [
        go.Scatter( # Surface
          x=surface[0], y=surface[1],
          mode="lines",
          line=dict(width=2, color="black"),
          # fill='tozeroy', 
          # fillcolor = 'white',
        ),
      ]

    return data
  def generateFigWithExtra(self, q_vec, z_vec, extra_frames, extra_surfaces, opacity_vec):

    # z = self.generateMeasurement(q)
    fig = go.Figure(
      data = self.generateFrameWithExtra(q_vec, z_vec, extra_frames, opacity_vec,extra_surfaces, show_tf=False, stationary=True, surface=self.surface),
      layout=go.Layout(
        showlegend=False,
        # autosize=True,
        width=1200,
        height=750,
        xaxis=dict(range=[-0.05, 1.05], autorange=False, zeroline=False),
        yaxis=dict(scaleanchor="x", scaleratio=1),
        # ticks='',
        paper_bgcolor='rgba(0.99,0.99,0.99,0)',
        plot_bgcolor='rgba(0.95,0.95,0.95,1)',
        # title_text="Contour following",
      ),
      layout_yaxis_range=[-0.001,0.6]
    )

    return fig
  
