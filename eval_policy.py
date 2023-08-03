import gym
import torch
import os
import numpy as np
# from gym.wrappers import Monitor
import glob
import re
import pandas as pd
import json
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from matplotlib import cm
from environments.Simulink_env.simulink_environment import SimulinkBasic
from environments.MVA4_env.mva4_enviroment import MVA4Basic

# Open figure in new window
matplotlib.use('QtAgg')

class Policy:
    def __init__(self,
                 env,
                 # env_name: str = 'name-of-experiment',
                 r_path: str = '/policy/model.pth'):
        self.env = env
        # self.env_name = env_name
        self.path = os.path.abspath(os.getcwd())

        # get latest experiment folder
        folder = self.path + '/data/' + self.env.env_name + '/*'
        # folder = folder.replace('_', '-')
        list_of_folder = glob.glob(folder)
        self.latest_folder = max(list_of_folder, key=os.path.getctime)

        # load torch model of current policy
        self.policy_path = self.path + r_path
        self.policy_model = torch.load(self.policy_path)

        # define policy network depth
        self.n = int(len(self.policy_model) / 2 - 1)

        # get experiment data
        self.tpd = None

        # define weight and bias list
        self.weight = list()
        self.bias = list()

        # load values from pytorch policy model into the weight and bias list
        for i in range(self.n - 1):
            weight_name = 'fc' + str(i) + '.weight'
            weight_tensor = self.policy_model[weight_name].cpu()
            self.weight.append(weight_tensor.numpy())

            bias_name = 'fc' + str(i) + '.bias'
            bias_tensor = self.policy_model[bias_name].cpu()
            self.bias.append(bias_tensor.numpy())

        weight_tensor = self.policy_model['last_fc.weight'].cpu()
        self.weight.append(weight_tensor.numpy())

        bias_tensor = self.policy_model['last_fc.bias'].cpu()
        self.bias.append(bias_tensor.numpy())

    def calc_action(self, s):
        """
        Calculate action from state according to loaded policy
        :param s:  np.array([], dtype=np.float32)
        :return:
        """
        # calculate activation of the first layer
        u = np.matmul(self.weight[0], s) + self.bias[0]
        for i in range(1, self.n):
            # use relu activation
            u = np.maximum(u, np.zeros(u.shape))
            # calculate activation of the current layer
            u = np.matmul(self.weight[i], u) + self.bias[i]
        # use hyperbolic tangent to get action bonds of [-1, 1]
        u = np.tanh(u)

        return u

    def plot_poicy(self):

        # Generate figure
        self.fig_p = plt.figure()
        # Generate line list for 1d plot
        self.line = [None] * self.env.n_actions
        # Generate active list for checkbox state
        self.active = [True] * self.env.n_observations
        # Generate observation vector for slider
        self.observation = np.zeros([self.env.n_observations])

        self.observation_range = np.zeros([self.env.n_observations, 2])
        self.observation_range[:, 0] = -1
        self.observation_range[:, 1] = 1
        # self.observation_range[1, :] = np.array([-0.01, 0.01])
        # Generate axis list for observation slider
        ax_slider = [None]*self.env.n_observations
        # Generate observation slider list
        slider = [None]*self.env.n_observations
        # Fill lists
        for i in range(self.env.n_observations):
            ax_slider[i] = self.fig_p.add_axes([0.4, 0.05+i*0.2/self.env.n_observations, 0.5, 0.05])
            slider[i] = Slider(
                ax=ax_slider[i],
                label=self.env.observation_name[i],
                valmin=self.observation_range[i, 0],
                valmax=self.observation_range[i, 1],
                valinit=self.observation[i],
            )

        # Generate checkbox
        ax_active = self.fig_p.add_axes([0.05, 0.05, 0.2, 0.15])
        labels = self.env.observation_name
        check = CheckButtons(ax_active, labels, self.active)

        # Read slider values
        for i in range(self.env.n_observations):
            self.observation[i] = slider[i].val

        # Function for checkbox update
        def func(label):
            # Read checkbox
            self.active = check.get_status()
            # 0 dimensional
            if self.env.n_observations-sum(self.active) == 0:
                # Delete axis of other dimensions
                if hasattr(self, 'ax_1d'):
                    self.fig_p.delaxes(self.ax_1d)
                    del self.ax_1d
                if hasattr(self, 'ax_2d'):
                    for i in range(self.env.n_actions):
                        self.fig_p.delaxes(self.ax_2d[i])
                    del self.ax_2d
                # Generate 0D axis
                self.ax_0d = self.fig_p.add_axes([0.15, 0.3, 0.8, 0.65])
                # Calculate action
                a = self.calc_action(self.observation)
                # Generate 0D plot
                self.bar = self.ax_0d.bar(self.env.action_name, a)
                self.ax_0d.set_ylim([-1, 1])
                self.ax_0d.set_ylabel('action')

            # 1 dimensional
            elif self.env.n_observations-sum(self.active) == 1:
                # Delete axis of other dimensions
                if hasattr(self, 'ax_0d'):
                    self.fig_p.delaxes(self.ax_0d)
                    del self.ax_0d
                if hasattr(self, 'ax_2d'):
                    for i in range(self.env.n_actions):
                        self.fig_p.delaxes(self.ax_2d[i])
                    del self.ax_2d
                # Generate 1D axis
                self.ax_1d = self.fig_p.add_axes([0.15, 0.3, 0.8, 0.65])
                # find fixed index
                for i in range(self.env.n_observations):
                    if not self.active[i]:
                        self.n_fixed = i
                # Generate matrix with 100 observation vectors
                o = np.zeros([100, self.env.n_observations])
                o[:, :] = self.observation[:]
                # Replace fixed index with x
                x = np.linspace(self.observation_range[self.n_fixed, 0], self.observation_range[self.n_fixed, 1], 100)
                o[:, self.n_fixed] = x
                # Calculate action
                a = np.zeros([100, self.env.n_observations])
                for i in range(100):
                    a[i, :] = self.calc_action(o[i, :])
                # Plot action over fixed axis
                for i in range(self.env.n_actions):
                    self.line[i] = self.ax_1d.plot(x, a[:, i])[0]
                    self.line[i].set_label(self.env.action_name[i])
                self.ax_1d.set_ylim([-1, 1])
                self.ax_1d.set_xlabel(self.env.observation_name[self.n_fixed])
                self.ax_1d.set_ylabel('action')
                self.ax_1d.legend()

            # 2 dimensional
            elif self.env.n_observations - sum(self.active) == 2:
                # Delete axis of other dimensions
                if hasattr(self, 'ax_0d'):
                    self.fig_p.delaxes(self.ax_0d)
                    del self.ax_0d
                if hasattr(self, 'ax_1d'):
                    self.fig_p.delaxes(self.ax_1d)
                    del self.ax_1d
                # Generate 2D axis list
                self.ax_2d = [None] * self.env.n_actions
                for i in range(self.env.n_actions):
                    self.ax_2d[i] = self.fig_p.add_axes([0.05+i*0.9/self.env.n_actions, 0.3, 0.9/self.env.n_actions, 0.65], projection='3d')
                # Find fix axis
                n_fixed_1 = None
                n_fixed_2 = None
                for i in range(self.env.n_observations):
                    if not self.active[i]:
                        if n_fixed_1 is None:
                            n_fixed_1 = i
                        else:
                            n_fixed_2 = i
                # Gen 2D meshgrid
                x = np.linspace(self.observation_range[n_fixed_1, 0], self.observation_range[n_fixed_1, 1], 100)
                y = np.linspace(self.observation_range[n_fixed_2, 0], self.observation_range[n_fixed_2, 1], 100)
                x, y = np.meshgrid(x, y)
                a = np.zeros([2, 100, 100])
                # Calc action for every mesh point
                for m in range(100):
                    for n in range(100):
                        observation_3d = self.observation
                        observation_3d[n_fixed_1] = x[m, n]
                        observation_3d[n_fixed_2] = y[m, n]
                        a[:, m, n] = self.calc_action(observation_3d)
                # Plot hypersurface
                for i in range(self.env.n_actions):
                    self.surf = self.ax_2d[i].plot_surface(x, y, a[i, :, :], cmap=cm.coolwarm, linewidth=0, antialiased=False)
                    self.ax_2d[i].set_xlabel(self.env.observation_name[n_fixed_1])
                    self.ax_2d[i].set_ylabel(self.env.observation_name[n_fixed_2])
                    self.ax_2d[i].set_zlabel(self.env.action_name[i])
                    self.ax_2d[i].set_title(self.env.action_name[i])
                self.fig_p.canvas.draw_idle()

            else:
                print('A 3D visualization still needs to be invented')

            # Update Plots
            update(None)

        def update(val):
            # Update function is called if slider is changed
            # read slider values
            for i in range(self.env.n_observations):
                self.observation[i] = slider[i].val
            # 0 dimensional
            if self.env.n_observations-sum(self.active) == 0:
                # Calculate action
                a = self.calc_action(self.observation)
                # Update plot
                for rect, h in zip(self.bar, a):
                    rect.set_height(h)
                # Delete old value texts in plot
                for txt in self.ax_0d.texts:
                    txt.set_visible(False)
                # Write new value texts
                for i, v in enumerate(a):
                    self.ax_0d.text(i-0.1, 1.05, str(round(v, 2)), color='blue', fontweight='bold')
                self.fig_p.canvas.draw_idle()
            # 1 dimensional
            elif self.env.n_observations-sum(self.active) == 1:
                # Calculate actions
                o = np.zeros([100, self.env.n_observations])
                o[:, :] = self.observation[:]
                x = np.linspace(-1, 1, 100)
                o[:, self.n_fixed] = x
                a = np.zeros([100, self.env.n_observations])
                for i in range(100):
                    a[i, :] = self.calc_action(o[i, :])
                # Update plot
                for i in range(self.env.n_actions):
                    self.line[i].set_ydata(a[:, i])
                self.fig_p.canvas.draw_idle()
            else:
                self.fig_p.canvas.draw_idle()

        # Run func for plotting
        func(None)

        # Define update as Slider update function
        for i in range(self.env.n_observations):
            slider[i].on_changed(update)

        # Define func as checkbox update funtion
        check.on_clicked(func)

        # Show plot
        plt.show(block=True)

    def policy2csv(self):
        dir = self.path + '/policy/'
        for f in os.listdir(dir):
            if not os.path.join(dir, f) == dir + 'model.pth':
                os.remove(os.path.join(dir, f))

        for i in range(self.n):
            weight_df = pd.DataFrame(self.weight[i])  # convert to a dataframe
            weight_matrix_path = self.path + '/policy/weight_' + str(i) + '.csv'
            weight_df.to_csv(weight_matrix_path, index=False)  # save to file

            bias_df = pd.DataFrame(self.bias[i])  # convert to a dataframe
            bias_matrix_path = self.path + '/policy/bias_' + str(i) + '.csv'
            bias_df.to_csv(bias_matrix_path, index=False)  # save to file

    def get_experiment_data(self):
        print(self.latest_folder)

        f = open(self.latest_folder + '/variant.json')
        data = json.load(f)
        print(data)

        file_name = self.latest_folder + '/progress.csv'
        tpd = pd.read_csv(file_name, header=0, index_col=0).to_dict()

        return tpd

    def plot_experiment(self):
        self.tpd = self.get_experiment_data()
        fig, axs = plt.subplots(3, 3)
        print(self.tpd.keys())
        epoch = self.get_array(self.tpd, 'epoch')

        eval_Returns_Mean = self.get_array(self.tpd, 'eval/Returns Mean')
        expl_Returns_Mean = self.get_array(self.tpd, 'expl/Returns Mean')
        eval_Rewards_Mean = self.get_array(self.tpd, 'eval/Rewards Mean')
        expl_Rewards_Mean = self.get_array(self.tpd, 'expl/Rewards Mean')
        trainer_Alpha = self.get_array(self.tpd, 'trainer/Alpha')
        trainer_Policy_Loss = self.get_array(self.tpd, 'trainer/Policy Loss')
        trainer_Q_Targets_Mean = self.get_array(self.tpd, 'trainer/Q Targets Mean')
        trainer_Q1_Predictions_Mean = self.get_array(self.tpd, 'trainer/Q1 Predictions Mean')
        trainer_Q2_Predictions_Mean = self.get_array(self.tpd, 'trainer/Q2 Predictions Mean')

        axs[0, 0].plot(epoch, eval_Returns_Mean)
        axs[0, 0].set_ylabel('$G_{eval}$')

        axs[0, 1].plot(epoch, expl_Returns_Mean)
        axs[0, 1].set_ylabel('$G_{expl}$')

        axs[0, 2].plot(epoch, trainer_Alpha)
        axs[0, 2].set_ylabel('$alpha$')

        axs[1, 0].plot(epoch, eval_Rewards_Mean)
        axs[1, 0].set_ylabel('$R_{eval}$')

        axs[1, 1].plot(epoch, expl_Rewards_Mean)
        axs[1, 1].set_ylabel('$R_{expl}$')

        axs[1, 2].plot(epoch, trainer_Policy_Loss)
        axs[1, 2].set_ylabel('$P_{loss}$')

        axs[2, 0].plot(epoch, trainer_Q1_Predictions_Mean)
        axs[2, 0].set_ylabel('$Q_1$')

        axs[2, 1].plot(epoch, trainer_Q2_Predictions_Mean)
        axs[2, 1].set_ylabel('$Q_2$')

        axs[2, 2].plot(epoch, trainer_Q_Targets_Mean)
        axs[2, 2].set_ylabel('$Q_{soll}$')

        axs[2, 0].set_xlabel('$Episode$')
        axs[2, 1].set_xlabel('$Episode$')
        axs[2, 2].set_xlabel('$Episode$')

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
        # figManager = plt.get_current_fig_manager()
        # figManager.window.showMaximized()
        plt.show(block=True)

    @staticmethod
    def get_array(tpd, name):
        return np.array(list(tpd[name].values()))


def get_video_path(env_name):
    path = os.path.abspath(os.getcwd())
    list_of_folder = glob.glob(path + '/video/' + env_name + '/*')

    if not list_of_folder:
        k = 0
    else:
        latest_folder = max(list_of_folder, key=os.path.getctime)
        latest_folder = latest_folder.replace(path + '/video/' + env_name, "")
        k = int(re.findall(r'\d+', str(latest_folder))[0])

    video_path = './video/' + env_name + '/' + str(k + 1)

    return video_path


env_name = "BipedalWalker-v3"
env_name = "SimulinkBasic"
relative_path = '/policy/model.pth'
# relative_path = '/saved_policies/' + env_name + '/1/model_good.pth'

video = False
save_csv = True
plot_learing = True
plot_policy = True
sim = True

if __name__ == '__main__':

    if env_name == "SimulinkBasic":
        # env = SimulinkBasic(if_render=False, eval_env=True, if_save_fiq=True)
        env = MVA4Basic(if_render=False, eval_env=True, if_save_fiq=True)
    else:
        if video:
            env = Monitor(gym.make(env_name), get_video_path(env_name), force=True)
        else:
            env = gym.make(env_name)

    policy = Policy(env=env, r_path=relative_path)

    if plot_learing:
        policy.plot_experiment()

    if plot_policy:
        policy.plot_poicy()

    if save_csv:
        policy.policy2csv()

    if sim:
        env.init_render()
        observation = env.reset()
        # env.render()
        i = 0
        finish = False
        done = False
        k = (60*60/20+1)

        while not finish:
            observation, reward, done, info = env.step(policy.calc_action(observation))
            env.render()
            if done or i == k:
                finish = True
                plt.show(block=True)



