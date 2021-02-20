import gym
import cartpole_environment
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import math


my_cartpole = cartpole_environment.CartPoleEnv_adv()
env = my_cartpole


# Generate universe variables

# Inputs
x = ctrl.Antecedent(np.arange(-env.x_threshold * 2, env.x_threshold * 2, 0.5), 'x')
x_dot = ctrl.Antecedent(np.arange(-500, 500, 0.5), 'x_dot')
theta = ctrl.Antecedent(np.arange(-env.theta_threshold_radians * 2, env.theta_threshold_radians * 2, 0.5), 'theta')
theta_dot = ctrl.Antecedent(np.arange(-500, 500, 0.5), 'theta_dot')

# Output
force = ctrl.Consequent(np.arange(-env.force_mag, env.force_mag, 0.5), 'force')

# Generate fuzzy membership functions

x['left'] = fuzz.trimf(x.universe, [-env.x_threshold * 2, -env.x_threshold * 2, -2])
x['middle'] = fuzz.trimf(x.universe, [-5, 0, 5])
x['right'] = fuzz.trimf(x.universe, [2, env.x_threshold * 2, env.x_threshold * 2])

x_dot['to_left'] = fuzz.trimf(x_dot.universe, [-500, -500, 0])
x_dot['static'] = fuzz.trimf(x_dot.universe, [-3, 0, 3])
x_dot['to_right'] = fuzz.trimf(x_dot.universe, [0, 500, 500])

theta['left'] = fuzz.trimf(theta.universe, [-env.theta_threshold_radians * 2, -env.theta_threshold_radians * 2, 0])
theta['vertical'] = fuzz.trimf(theta.universe, [-20 * 2 * math.pi / 360, 0, 20 * 2 * math.pi / 360])
theta['right'] = fuzz.trimf(theta.universe, [0, env.theta_threshold_radians * 2, env.theta_threshold_radians * 2])

theta_dot['to_left'] = fuzz.trimf(theta_dot.universe, [-500, -500, 0])
theta_dot['static'] = fuzz.trimf(theta_dot.universe, [-3, 0, 3])
theta_dot['to_right'] = fuzz.trimf(theta_dot.universe, [0, 500, 500])

force['to_left'] = fuzz.trimf(force.universe, [-env.force_mag, -env.force_mag, 0])
force['static'] = fuzz.trimf(force.universe, [-env.force_mag, 0, env.force_mag])
force['to_right'] = fuzz.trimf(force.universe, [0, env.force_mag, env.force_mag])

# Generate fuzzy rules
rule1 = ctrl.Rule(x['left'] & x_dot['to_left'], force['to_right'])
rule2 = ctrl.Rule(x['right'] & x_dot['to_right'], force['to_left'])
rule3 = ctrl.Rule(theta['left'] & theta_dot['to_left'], force['to_left'])
rule4 = ctrl.Rule(theta['right'] & theta_dot['to_right'], force['to_right'])


my_cartpole_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
my_cartpole_ctrl_sim = ctrl.ControlSystemSimulation(my_cartpole_ctrl)

MAX_EPISODES = 20
MAX_EPISODES_STEPS = 200
total_reward = 0

for i_episode in range(MAX_EPISODES):
    observation = env.reset()
    episode_reward = 0
    for timestep in range(MAX_EPISODES_STEPS):
        env.render()
        print(observation)
        my_cartpole_ctrl_sim.inputs({'x': observation[0],
                                     'x_dot': observation[1],
                                     'theta': observation[2],
                                     'theta_dot': observation[3]})
        
        my_cartpole_ctrl_sim.compute()
        action = my_cartpole_ctrl_sim.output['force']
        observation, reward, done, info = env.step(action)
        episode_reward += reward
        if done:
            print("Episode finished after {} timesteps".format(timestep+1))
            break
    print("Episode reward is:", episode_reward)
    total_reward += episode_reward

print("Average reward is:", total_reward / MAX_EPISODES)