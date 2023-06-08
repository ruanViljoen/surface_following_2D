import matplotlib.pyplot as plt
import numpy as np

def plot_experiment_dashboard(all_logs, experiment_name):
    """
    Plots a dashboard of experiment data.

    Args:
        all_logs (dict): Dictionary containing logs for all experiments.
        experiment_name (str): Name of the experiment to plot.

    Returns:
        None
    """
    # Extract the relevant data from all_logs based on experiment_name
    df_selected = all_logs[experiment_name]
    
    # Extract data arrays from df_selected
    q = np.array(list(df_selected["q"]))  
    q_dot = np.array(list(df_selected["dq"]))  
    q_ddot = np.array(list(df_selected["ddq"]))  
    e = np.array(list(df_selected["task_translation_error"]))
    t_vec = np.array(list(df_selected['t'])).T
    
    # Create a new figure
    fig = plt.figure(figsize=(25, 10))
    
    # Plot joint positions
    ax = fig.add_subplot(2, 2, 1)
    ax.plot(t_vec, q)
    ax.set_xlabel('time [s]')
    ax.set_ylabel('joint positions')
    ax.legend(['joint 0 [m]', 'joint 1 [rad]', 'joint 2 [rad]', 'joint 3 [rad]'])
    ax.grid()
    
    # Plot joint velocities
    ax = fig.add_subplot(2, 2, 2)
    ax.plot(t_vec, q_dot)
    ax.set_xlabel('time [s]')
    ax.set_ylabel('joint velocities')
    ax.legend(['joint 0 [m/s]', 'joint 1 [rad/s]', 'joint 2 [rad/s]', 'joint 3 [rad/s]'])
    ax.grid()
    
    # Plot joint accelerations
    ax = fig.add_subplot(2, 2, 3)
    ax.plot(t_vec, q_ddot)
    ax.set_xlabel('time [s]')
    ax.set_ylabel('joint accelerations')
    ax.legend(['joint 0 [m/s$^2$]', 'joint 1 [rad/s$^2$]', 'joint 2 [rad/s$^2$]', 'joint 3 [rad/s$^2$]'])
    ax.grid()
    
    # Uncomment the following lines if you want to plot task translation error
    # ax = fig.add_subplot(3, 2, 4)
    # ax.plot(t_vec, e)
    # ax.set_xlabel('time [s]')
    
    # Adjust the layout of subplots for better spacing
    plt.tight_layout()

def compare_tracking_errors(all_logs, experiment_names):
    """
    Compares the tracking errors of different experiments.

    Args:
        all_logs (dict): Dictionary containing logs for all experiments.
        experiment_names (list): List of experiment names to compare.

    Returns:
        None
    """
    # Create a new figure with subplots
    fig = plt.figure(figsize=(25, 10))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
  
    # Iterate over experiment names
    for experiment_name in experiment_names:
        # Extract tracking error data for the current experiment
        e_tr = np.array(list(all_logs[experiment_name]['task_translation_error']))
        e_o = np.array(list(all_logs[experiment_name]['task_orientation_error']))
        e_ps = np.array(list(all_logs[experiment_name]['task_progress_speed_error']))
        t_vec = np.array(list(all_logs[experiment_name]['t']))
        
        # Plot tracking error for x-coordinate
        ax1.plot(t_vec, e_tr[:, 0], label=experiment_name)
        # Plot tracking error for y-coordinate
        ax2.plot(t_vec, e_tr[:, 1], label=experiment_name)
        # Plot tracking error for orientation
        ax3.plot(t_vec, e_o[:, 0], label=experiment_name)
        # Plot tracking error for progress speed
        ax4.plot(t_vec, e_ps[:, 0], label=experiment_name)
  
    # Set grid and labels for each subplot
    ax1.grid()
    ax1.set_xlabel('time [s]')
    ax1.set_ylabel('$x$ tracking error [m]')
    ax1.legend()
    ax2.grid()
    ax2.set_xlabel('time [s]')
    ax2.set_ylabel('$y$ tracking error [m]')
    ax2.legend()
    ax3.grid()
    ax3.set_xlabel('time [s]')
    ax3.set_ylabel('$\Theta$ tracking error [rad]')
    ax3.legend()
    ax4.grid()
    ax4.set_xlabel('time [s]')
    ax4.set_ylabel('progress speed tracking error [m/s]')
    ax4.legend()
  
def compare_outputs_box_whiskers(all_logs, experiment_names, output_name, log=True):
    """
    Compares the outputs of different experiments using box-and-whisker plots.

    Args:
        all_logs (dict): Dictionary containing logs for all experiments.
        experiment_names (list): List of experiment names to compare.
        output_name (str): Name of the output to compare.
        log (bool, optional): Flag indicating whether to use logarithmic scale on the y-axis. Defaults to True.

    Returns:
        None
    """
    # Collect the output data for each experiment into a list
    time_taken_list = [np.array(list(all_logs[experiment_name][output_name])).flatten() for experiment_name in experiment_names]
  
    # Create a new figure
    plt.figure(figsize=(15, 5))
  
    # Create a box-and-whisker plot of the output data
    plt.boxplot(time_taken_list, labels=experiment_names)
  
    # Set y-axis scale to logarithmic if log=True
    if log:
        plt.yscale('log')
  
    # Set the y-axis label to the output_name
    plt.ylabel(output_name)

def compare_outputs(all_logs,experiment_names, output_name):
  
  t_vec = np.array(list(all_logs[experiment_names[0]]['t'])).T
  
  # Number of elements of task
  nt = np.array(list(all_logs[experiment_names[0]][output_name])).shape[1]
  
  fig = plt.figure(figsize=(20,int(5*nt)))
  
  axes = [ fig.add_subplot(nt,1,i+1) for i in range(nt) ] 
  
  for experiment_name in experiment_names:
    
    e = np.array(list(all_logs[experiment_name][output_name]))
    t_vec = np.array(list(all_logs[experiment_name]['t'])).T
    for i in range(nt):
      axes[i].plot(t_vec,e[:,i],label=experiment_name)
  
  for i in range(nt):
    axes[i].legend()
    axes[i].grid()
    axes[i].set_xlabel('time [s]')
    axes[i].set_ylabel(output_name+'_'+str(i))
    
    
  # axes[0].set_ylabel('$\Theta$ [rad]')
  # axes[0].set_ylabel('x [m]')
  # axes[1].set_ylabel('y [m]')
  
def plot_animation(simulator,all_logs,experiment_name,h,stride=1):
  df_selected = all_logs[experiment_name]
  q_vec = np.array(list(df_selected['q']))[::stride]
  q_total_vec = [
    [
      q_vec[i],
    ] for i in range(len(q_vec))
  ]
  
  path_vec = np.array(list(df_selected['path']))[::stride]
  z_vec = [simulator.generateMeasurement(q) for q in q_vec]
  z_total_vec = [
    [
      z_vec[i],
    ] for i in range(len(z_vec))
  ]

  w_est_vec = np.array(list(df_selected['w']))[::stride]
  predicted_traj_vec = np.array([generate_predicted_trajectory(w,h) for w in w_est_vec])

  # Generate basis functions
  basis_functions = []
  nw = w_est_vec.shape[1]
  for i in range(nw):
    selection_matrix = np.zeros((nw,nw))
    selection_matrix[i,i]=1
    # if i <= 10:
    basis_functions += [np.array([generate_predicted_trajectory(selection_matrix@w.reshape(-1,1),h) for w in w_est_vec])]
    
  x_window_vec = list(df_selected['x_window'])[::stride]
  x_MPC_window_vec = list(df_selected['x_MPC_window'])[::stride]
  MPC_horizon_length = x_MPC_window_vec[0].shape[1]
  
  extra_surfaces = [
    [
      [
        x_window_vec[i],
        simulator.f_surface(x_window_vec[i]).full().flatten()
      ],
      [
        x_MPC_window_vec[i].flatten(),
        h(x_MPC_window_vec[i].reshape(1,-1),np.repeat(w_est_vec[i].reshape(-1,1),MPC_horizon_length,axis=1)).full().flatten()
        # simulator.f_surface(x_MPC_window_vec[i]).full().flatten()
      ],
      predicted_traj_vec[i],
    ] for i in range(len(x_MPC_window_vec))
  ]

  # Add basis functions to extra_surfaces
  for i in range(len(extra_surfaces)):
    for j in range(len(basis_functions)):
      extra_surfaces[i] += [basis_functions[j][i]]
    
  opacity_vec = [1]
  
  return simulator.generateAnimation(q_total_vec,z_total_vec,opacity_vec,path_vec,extra_surfaces)

def generate_predicted_trajectory(w,h):
  x_vec = np.linspace(-0,1,100)
  z_vec = []
  for x,i in zip(x_vec, range(x_vec.shape[0])):
    z = h(x,w)
    z_vec += [z.full().flatten()[0]]
  return x_vec, np.array(z_vec)
  
def compare_animation(all_logs,experiment_names,stride=1):
  
  experiment_name = experiment_names[0]
  
  df_selected = all_logs[experiment_name]
  q_vec = np.array(list(df_selected['q']))[::stride]
  
  q_total_vec = [
    [
      q_vec[i],
      np.array(list(all_logs[experiment_names[-1]]['q']))[::stride][i]
    ] for i in range(len(q_vec))
  ]
  
  path_vec = np.array(list(df_selected['path']))[::stride]
  
  z_vec = [simulator.generateMeasurement(q) for q in q_vec]
  z_total_vec = [
    [
      z_vec[i],
      0
    ] for i in range(len(z_vec))
  ]
  

  w_est_vec = np.array(list(df_selected['w']))[::stride]
  predicted_traj_vec = np.array([generate_predicted_trajectory(w,h) for w in w_est_vec])
  x_window_vec = list(df_selected['x_window'])[::stride]
  
  extra_surfaces = [
    [
      predicted_traj_vec[i],
      [
        x_window_vec[i],
        simulator.f_surface(x_window_vec[i]).full().flatten()
      ]
    ] for i in range(len(x_window_vec))
  ]
  
  opacity_vec = [0.5,0.5]
  
  return simulator.generateAnimation(q_total_vec,z_total_vec,opacity_vec,path_vec,extra_surfaces)

def save_animation(all_logs,experiment_name,stride):
  plot_animation(experiment_name,stride).write_html(experiment_name+".html")


def compare_sum_sqr_outputs(all_logs,experiment_names, output_name):
  
  t_vec = np.array(list(all_logs[experiment_names[0]]['t'])).T
  
  # Number of elements of task
  nt = np.array(list(all_logs[experiment_names[0]][output_name])).shape[1]
  
  fig = plt.figure(figsize=(10,int(3*(nt+1))))
  
  axes = [ fig.add_subplot(nt+1,1,i+1) for i in range(nt+1) ] 
  
  for experiment_name in experiment_names:
    
    y = np.array(list(all_logs[experiment_name][output_name]))
    
    axes[0].plot(t_vec,np.cumsum(np.diag(y@y.T)),label=experiment_name)
    
    for i in range(nt):
      axes[i+1].plot(t_vec,np.cumsum(y[:,i]**2),label=experiment_name)
  
  for i in range(1,nt+1):
    axes[i].legend()
    axes[i].grid()
    axes[i].set_xlabel('time [s]')
    axes[i].set_ylabel(output_name+'_'+str(i-1))
    
  axes[0].legend()
  axes[0].grid()
  axes[0].set_xlabel('time [s]')
  axes[0].set_ylabel(output_name)    
  
  fig.tight_layout()