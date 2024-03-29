# ---------------------------------------------------------------+
#
#   Albert Negura
#   2-Dimensional Particle Swarm Optimization (PSO) with Python
#   February, 2021
#
# ---------------------------------------------------------------+
# --- IMPORT DEPENDENCIES----------------------------------------+
# mathematics / algorithm imports
import math
import numpy as np
from functools import partial
# matplotlib for plotting
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, RadioButtons
matplotlib.use("TkAgg")
# config parser for .ini
import configparser


# --- PSO CLASS--------------------------------------------------+
def rand_cmap(nlabels):
    from matplotlib.colors import LinearSegmentedColormap
    import colorsys
    import numpy as np

    # Generate color map for bright colors, based on hsv
    randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                      np.random.uniform(low=0.2, high=1),
                      np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

    # Convert HSV list to RGB
    randRGBcolors = []
    for HSVcolor in randHSVcolors:
        randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

    random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    return random_colormap


class PSO():
    # config-adjustable parameters
    swarmsize = None
    iterations = None
    omega = None  # inertia
    c1 = None  # cognitive constant (lambda 1)
    c2 = None  # social constant (lambda 2)
    T1 = None
    T2 = None
    CONVERGENCE = None
    PROCESSES = None # Se PROCESSES è maggiore di 1, allora viene utilizzato il calcolo parallelo per calcolare il funzionale
    mp_pool = None

    # function selector (6 implemented functions)
    function = None
    lower_bounds = None
    upper_bounds = None
    goal = None

    # plotting lists
    x_hist = None
    v_hist = None
    avg_cost_function = None
    min_cost_function = None

    # scale_factor = None

    def __init__(self, mode='config', swarmsize=100, iterations=100, omega=0.5, c1=0.5, c2=0.5,
                 T1=1e-10, T2=1e-10, CONVERGENCE=False, PROCESSES=1, function=0):
        if mode != 'config':
            self.swarmsize = swarmsize
            self.iterations = iterations
            self.omega = omega
            self.c1 = c1
            self.c2 = c2
            self.T1 = T1
            self.T2 = T2
            self.CONVERGENCE = CONVERGENCE
            self.PROCESSES = PROCESSES

            self.function = function
        else:

            config = configparser.ConfigParser()
            config.read('config.ini')
            pso_config = config['pso']

            self.swarmsize = pso_config.getint("swarm_size")
            self.iterations = pso_config.getint("maximum_iterations")
            self.omega = pso_config.getfloat("inertia")
            self.c1 = pso_config.getfloat("cognitive_constant")
            self.c2 = pso_config.getfloat("social_constant")
            self.T1 = pso_config.getfloat("step_convergence_threshold")
            self.T2 = pso_config.getfloat("value_convergence_threshold")
            self.CONVERGENCE = pso_config.getboolean("converge_early")
            self.PROCESSES = pso_config.getint("PROCESSES")

            self.function = config['functions'].getint("function_selection")

        self.x_hist = np.zeros((self.iterations, self.swarmsize, 2))
        self.v_hist = np.zeros((self.iterations, self.swarmsize, 2))
        self.avg_cost_function = np.zeros((self.iterations))
        self.min_cost_function = np.zeros((self.iterations))
        self.ani_list = [animation.FuncAnimation, animation.FuncAnimation]

        self.lower_bounds = [0, 0]
        self.upper_bounds = [10, 10]

        # initialize upper, lower bound and global minimum based on the function selector
        if self.function == 0:
            self.lower_bounds = [0, 0]
            self.upper_bounds = [10, 10]
            self.goal = [7.917, 7.917]
        elif self.function == 1:
            self.lower_bounds = [-4.5, -4.5]
            self.upper_bounds = [4.5, 4.5]
            self.goal = [3, 0.5]
        elif self.function == 2:
            self.lower_bounds = [-2 * math.pi, -2 * math.pi]
            self.upper_bounds = [2 * math.pi, 2 * math.pi]
            self.goal = [[4.70104, 3.15294], [-1.58214, -3.13024]]
        elif self.function == 3:
            self.lower_bounds = [-5.2, -5.2]
            self.upper_bounds = [5.2, 5.2]
            self.goal = [0, 0]
        elif self.function == 4:
            self.lower_bounds = [-5.12, -5.12]
            self.upper_bounds = [5.12, 5.12]
            self.goal = [0, 0]
        elif self.function == 5:
            self.lower_bounds = [-5, -5]
            self.upper_bounds = [10, 10]
            self.goal = [1, 1]

        # self.scale_factor = np.abs((np.max(self.upper_bounds) - np.min(self.lower_bounds))) * 2

    # --- COST FUNCTION----------------------------------------------+
    def error(self, x):
        x1 = x[0]
        x2 = x[1]
        if self.function == 0:  # alpine n.2
            return -(np.sqrt(np.abs(x1)) * np.sin(x1) * np.sqrt(np.abs(x2)) * np.sin(x2))
        elif self.function == 1:  # beale
            return (1.5 - x1 + x1 * x2) ** 2 + (2.25 - x1 + x1 * x2 ** 2) ** 2 + (2.625 - x1 + x1 * x2 ** 3) ** 2
        elif self.function == 2:  # bird
            return np.sin(x1) * np.exp((1 - np.cos(x2)) ** 2) + np.cos(x2) * np.exp((1 - np.sin(x1)) ** 2)
        elif self.function == 3:  # drop-wave
            return -(1 + np.cos(12 * np.sqrt(x1 ** 2 + x2 ** 2))) / (0.5 * (x1 ** 2 + x2 ** 2) + 2)
        elif self.function == 4:  # rastrigin
            return 20 + (x1 ** 2 - 10 * np.cos(math.pi * 2 * x1)) + (x2 ** 2 - 10 * np.cos(math.pi * 2 * x2))
        elif self.function == 5:  # rosenbrock a=0,b=1
            return (x2 - x1 ** 2) ** 2 + x1 ** 2
        else:
            return - (np.sqrt(x1) * np.sin(x1) * np.sqrt(x2) * np.sin(x2))

    def function_of(self, x, y):
        return self.error([x, y])

    def error_plot(self, values):
        z = np.zeros(values.shape[0])
        for i in range(values.shape[0]):
            val = values[i]
            z[i] = self.error(val)

        return z


    def particle_swarm_optimization(self):
        # local copies of bounds for shorter calls
        lb = np.array(self.lower_bounds.copy()) # carica il valore del limite inferiore delle possibili soluzioni
        ub = np.array(self.upper_bounds.copy()) # carica il valore del limite superiore delle possibili soluzioni
        # Assert interrompe l'esecuzione con un messaggio di errore se la condizione è falsa
        assert np.all(ub > lb), 'All upper bound values must be greater than the corresponding lower bound values'

        # set lower and upper bounds to velocities based on position bounds
        upper_bound_velocity = np.abs(ub - lb)
        lower_bound_velocity = -upper_bound_velocity

        # objective contiene l'errore, ovvero il valore del funzionale
        objective = self.error

        # Se PROCESSES è maggiore di uno, allora predispone il calcolo parallelo
        if self.PROCESSES > 1:
            import multiprocessing
            # Inizilizza un numero di processi pari a PROCESSES
            mp_pool = multiprocessing.Pool(self.PROCESSES)

        # initialize particle positions randomly in the function bounds
        positions = np.random.rand(self.swarmsize, 2)  # particle position
        positions = lb + positions * (ub - lb)

        # initialize a few arrays
        best_positions = np.zeros_like(positions)  # best known position per particle
        objectives = np.zeros(self.swarmsize)  # objective function value per particle
        best_objectives = np.ones(self.swarmsize) * np.inf  # best particle position objective function value
        best_swarm_positions = []
        best_swarm_objective = np.inf  # best swarm position

        # calculate objectives for each particles
        # se PROCESSES è maggiore di 1, calcola l'errore in parallelo
        if self.PROCESSES > 1:
            objectives = np.array(self.mp_pool.map(objective, positions))
        else:
            for i in range(self.swarmsize):
                objectives[i] = objective(positions[i, :])  # calculate objective function

        i_update = objectives < best_objectives  # seleziona le particelle a cui aggiornare il personal best
        best_positions[i_update, :] = positions[i_update, :].copy() # aggiorna la posizione del personal best
        best_objectives[i_update] = objectives[i_update]  # aggiorna il valore del personal best

        # index of best particle
        i_min = np.argmin(best_objectives)
        if best_objectives[i_min] < best_swarm_objective:  # if the best particle is in a better position than all other particles
            best_swarm_objective = best_objectives[i_min]
            best_swarm_positions = best_positions[i_min, :].copy()  # best known swarm position
        else:
            best_swarm_positions = positions[0, :].copy()  # best known swarm position

        # calculate initial velocity vector
        velocities = lower_bound_velocity + np.random.rand(self.swarmsize, 2) * (
                upper_bound_velocity - lower_bound_velocity)

        # Ciclo principale dell'algoritmo
        it = 1  # Contatore iterazioni
        while it <= self.iterations:
            # add position/velocity of all particles to history array for plotting
            # registra la storia delle posizioni e delle velocità delle particelle, per il plotting
            self.x_hist[it - 1] = np.array(positions)
            self.v_hist[it - 1] = np.array(velocities)
            # update velocity vector with slight randomization to approach minimum
            rp = np.random.uniform(size=(self.swarmsize, 2))  # variabile aleatoria per il personal best
            rg = np.random.uniform(size=(self.swarmsize, 2))  # variabile aleatoria per il global best
            # calcolo delle velocità delle particelle
            velocities = self.omega * velocities + self.c1 * rp * (best_positions - positions) + self.c2 * rg * (
                    best_swarm_positions - positions)
            # update position vector
            positions = positions + velocities

            # prevent out of bounds
            # lower_mask e upper_mask sono true se le posizioni delle particelle seono
            # rispettivamente più basse o più alte dei limiti (bound) dello
            # spazioe di ricerca
            lower_mask = positions < lb
            upper_mask = positions > ub

            # Se le posizioni superano i limiti, le particelle vengono riposizionate sul valore del limite superato
            # In sostanza vengono riposizionate sui bordi dello spazio di ricerca.
            positions = positions * (~np.logical_or(lower_mask, upper_mask)) + lb * lower_mask + ub * upper_mask

            # update objective function
            if self.PROCESSES > 1:
                objectives = np.array(self.mp_pool.map(objective, positions))
            else:
                for i in range(self.swarmsize):
                    objectives[i] = objective(positions[i, :])

            # store best position
            i_update = objectives < best_objectives
            best_positions[i_update, :] = positions[i_update, :].copy()
            best_objectives[i_update] = objectives[i_update]

            # compare swarm best position with global best position
            i_min = np.argmin(best_objectives)
            self.min_cost_function[it - 1] = best_objectives[i_min]  # min cost function for plotting
            self.avg_cost_function[it - 1] = np.average(best_objectives)  # average cost function for plotting
            if best_objectives[i_min] < best_swarm_objective:

                best_particle_position = best_positions[i_min, :].copy()
                stepsize = np.sqrt(np.sum((best_swarm_positions - best_particle_position) ** 2))

                # converge early
                # if swarm objective change is too small
                if self.CONVERGENCE and np.abs(best_swarm_objective - best_objectives[i_min]) <= self.T2:
                    self.iterations = it
                    self.x_hist = self.x_hist[:it]
                    self.v_hist = self.v_hist[:it]
                    self.min_cost_function = self.min_cost_function[:it]
                    self.avg_cost_function = self.avg_cost_function[:it]
                # else if swarm best position change is too small
                elif self.CONVERGENCE and stepsize <= self.T1:
                    self.iterations = it
                    self.x_hist = self.x_hist[:it]
                    self.v_hist = self.v_hist[:it]
                    self.min_cost_function = self.min_cost_function[:it]
                    self.avg_cost_function = self.avg_cost_function[:it]
                # else do not converge early and iterate again
                else:
                    best_swarm_positions = best_particle_position.copy()
                    best_swarm_objective = best_objectives[i_min]
            it += 1

    # --- PLOTTING---------------------------------------------------+
    def plot_all(self):
        from multiprocessing import Process
        fig, ((ax1,ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=False, sharey=False)
        fig.set_figheight(15)
        fig.set_figwidth(15)
        p1 = Process(pso.animate2D(self.min_cost_function, "Min", fig, ax1, 1))
        p1.start()
        p2 = Process(pso.animate2D(self.avg_cost_function, "Average", fig, ax2, 2))
        p2.start()
        p3 = Process(pso.animate_contour(self.x_hist, self.v_hist, fig, ax3))
        p3.start()
        p4 = Process(pso.animate3D(self.x_hist, self.v_hist, fig, ax4))
        p4.start()
        plt.show()

    def animate2D(self, data_used, label, fig, ax1, ax):
        self.data = data_used.copy()
        self.stop = np.size(self.data)
        ax1.set(xlim=[0, self.stop], ylim=[np.min(self.data), np.max(self.data)])
        ax1.tick_params(axis='x', labelbottom=False)
        ax1.tick_params(axis='y', labelleft=False)
        indices = np.linspace(0, self.stop, self.stop - 1)
        axis = fig.add_subplot(2,2,ax)
        #ax.xlabel.set_text('Iterations')
        #ax.ylabel.set_text('Cost')
        axis.title.set_text(label + ' Cost Function')
        axis.plot(self.data, lw=3)
        #line, = axis.plot([], [], lw=3)
        #self.ani_list[ax-1] = (animation.FuncAnimation(fig, self.animate, frames=self.iterations,fargs=[indices, self.data, line, ax], interval=20, blit=True))

    def animate(self, i, x, y, line, ax):
        if i >= self.stop - 1:
            self.ani_list[ax-1].event_source.stop()
        line.set_data(x[:i], y[:i])
        #line.axes.axis([0, np.size(self.data), np.min(self.data), np.max(self.data)])
        return line,

    def animate_contour(self, positions, velocities, fig, ax):
        self.xs = positions.copy()
        self.vs = velocities.copy()

        self.stop = self.xs.shape[0]
        ax.set(xlim=[np.min(self.lower_bounds), np.max(self.upper_bounds)],
                            ylim=[np.min(self.lower_bounds), np.max(self.upper_bounds)])

        if np.max(self.upper_bounds) > 0 and np.min(self.lower_bounds) < 0:
            x = np.arange(np.min(self.lower_bounds) * 2, np.max(self.upper_bounds) * 2, 0.05)
            y = np.arange(np.min(self.lower_bounds) * 2, np.max(self.upper_bounds) * 2, 0.05)
        elif np.min(self.lower_bounds) < 0 and np.max(self.upper_bounds) < 0:
            x = np.arange(np.min(self.lower_bounds), 0 - np.max(self.upper_bounds), 0.05)
            y = np.arange(np.min(self.lower_bounds), 0 - np.max(self.upper_bounds), 0.05)
        elif np.min(self.lower_bounds) > 0 and np.max(self.upper_bounds) > 0:
            x = np.arange(abs(np.min(self.lower_bounds)) + np.min(self.lower_bounds), 2 * np.max(self.upper_bounds),
                          0.05)
            y = np.arange(abs(np.min(self.lower_bounds)) + np.min(self.lower_bounds), 2 * np.max(self.upper_bounds),
                          0.05)
        else:
            x = np.arange(2 * np.min(self.lower_bounds), abs(np.max(self.upper_bounds)) + np.max(self.upper_bounds),
                          0.05)
            y = np.arange(2 * np.min(self.lower_bounds), abs(np.max(self.upper_bounds)) + np.max(self.upper_bounds),
                          0.05)

        X, Y = np.meshgrid(x, y)
        zs = np.array(self.function_of(np.ravel(X), np.ravel(Y)))
        Z = zs.reshape(X.shape)

        self.CS = ax.contour(X, Y, Z, levels = 20, cmap='viridis')
        ax.title.set_text("2D Contour Plot of Objective Function")

        Xs = self.xs[0]
        x_Xs = Xs[:, 0]
        y_Xs = Xs[:, 1]
        Vs = self.vs[0]
        x_Vs = Vs[:, 0]
        y_Vs = Vs[:, 1]

        cmap = rand_cmap(self.swarmsize)
        if len(self.goal) == 2:
            goal_scatter = ax.scatter(self.goal[0], self.goal[1], s=self.swarmsize * 10, marker="x")
        else:
            goal_x = self.goal[:, 0]
            goal_y = self.goal[:, 1]
            goal_scatter = ax.scatter(goal_x, goal_y, s=self.swarmsize * 10, marker="x")
        scatters = ax.scatter(x_Xs, y_Xs, c=[i for i in range(self.swarmsize)], cmap=cmap, marker="o", vmin=0,
                                    vmax=self.swarmsize)
        # self.contour_vectors = self.ax2.quiver(x_Xs, y_Xs, x_Vs, y_Vs, scale=50)
        lines = []
        for i in range(self.swarmsize):
            line = ax.plot(self.xs[0, i, 0], self.xs[0, i, 1], c=cmap(i), alpha=0.3)
            lines.append(line)
        self.ani2 = animation.FuncAnimation(fig, self.animate2, frames=self.iterations, fargs=[scatters, lines],
                                            interval=100, blit=False, repeat=True)

    def animate2(self, i, scatters, lines):
        # global contour_vectors
        plot_data = self.xs[i]
        v_plot_data = self.vs[i]

        # self.contour_vectors.remove()
        scatters.set_offsets(plot_data)
        if i > 5:
            for lnum, line in enumerate(lines):
                data = self.xs[i - 5:i, lnum, :]
                line[0].set_data(data[:, 0], data[:, 1])
        # self.contour_vectors = self.ax2.quiver(plot_data[:, 0], plot_data[:, 1], v_plot_data[:, 0], v_plot_data[:, 1],scale=50)
        return scatters,

    def animate3D(self, positions, velocities, fig, ax):
        self.xs = positions.copy()
        self.vs = velocities.copy()

        ax = fig.add_subplot(2,2,4, projection='3d')
        x = np.arange(np.min(self.lower_bounds), np.max(self.upper_bounds), 0.05)
        y = np.arange(np.min(self.lower_bounds), np.max(self.upper_bounds), 0.05)
        X, Y = np.meshgrid(x, y)
        zs = np.array(self.function_of(np.ravel(X), np.ravel(Y)))
        Z = zs.reshape(X.shape)

        ax.plot_surface(X, Y, Z, cmap='gray', edgecolor='none', alpha=0.2)
        ax.title.set_text("3D Plot of Objective Function")

        self.stop = self.xs.shape[0]
        # self.scale_factor /= self.stop
        Xs = self.xs[0]
        x_Xs = Xs[:, 0]
        y_Xs = Xs[:, 1]
        z_Xs = self.error_plot(Xs[:, :])
        Vs = self.vs[0]
        x_Vs = Vs[:, 0]  # * self.scale_factor
        y_Vs = Vs[:, 1]  # * self.scale_factor
        z_Vs = self.error_plot(Vs[:, :])  # * self.scale_factor

        if len(self.goal) == 2:
            goal_z = self.error_plot(np.array([self.goal]))
            goal_scatter = ax.scatter(self.goal[0], self.goal[1], goal_z, s=self.swarmsize * 10, marker="x")
        else:
            goal_x = self.goal[:, 0]
            goal_y = self.goal[:, 1]
            goal_z = self.error_plot(np.array(self.goal))
            goal_scatter = ax.scatter(goal_x, goal_y, goal_z, s=self.swarmsize * 10, marker="x")

        cmap = rand_cmap(self.swarmsize)
        scatters = ax.scatter(x_Xs, y_Xs, z_Xs, c=[i for i in range(self.swarmsize)], cmap=cmap, marker="o",
                                    vmin=0, vmax=self.swarmsize)
        # self.vectors = self.ax3.quiver(x_Xs, y_Xs, z_Xs, x_Vs, y_Vs, z_Vs)
        lines = []
        for i in range(self.swarmsize):
            line = ax.plot(self.xs[0, i, 0], self.xs[0, i, 1], z_Xs[i], c=cmap(i), alpha=0.5)
            lines.append(line)

        self.ani3 = animation.FuncAnimation(fig, self.animate3, frames=self.iterations, fargs=[scatters, lines],
                                            interval=100,
                                            blit=False, repeat=True)
    def animate3(self, i, scatters, lines):
        # global vectors, scale_factor
        plot_data = self.xs[i]
        v_plot_data = self.vs[i]
        z_Xs = self.error_plot(plot_data[:])

        # self.vectors.remove()
        if i > 5:
            for lnum, line in enumerate(lines):
                data = self.xs[i - 5:i, lnum, :]
                function_data = self.error_plot(data)
                line[0].set_data(data[:, 0], data[:, 1])
                line[0].set_3d_properties(function_data)
        scatters._offsets3d = (plot_data[:, 0], plot_data[:, 1], z_Xs)
        # self.vectors = self.ax3.quiver(plot_data[:, 0], plot_data[:, 1], z_Xs,v_plot_data[:, 0] * self.scale_factor, v_plot_data[:, 1] * self.scale_factor,z_Xs * self.scale_factor)
        return scatters,


# __name__ è una variabile di sistema che Python inizializza prima di eseguire il codice. Se il sorgente viene
# esegiuto come main, ovvero come punto di entrata, Pyhton inzializza la variabile __name__ con '__main__'. 
# Questo serve, ad esempio, a evitare che parte del codice NON venga eseguito se il sorgente viene importato e 
# non utilizzato come punto di ingresso
if __name__ == '__main__':

    # crea l'oggetto PSO
    pso = PSO()
    
    # esegue l'algoritmo PSO
    pso.particle_swarm_optimization()

    # visualizza l'animazione del PSO creata durante l'esecuzione dell'algortimo
    pso.plot_all()


