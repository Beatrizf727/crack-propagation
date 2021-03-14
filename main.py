import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import math
import time

# -------------- Constants

C = .1
n = 1

# -------------- Functions

def load_to_tension(N: float):
    return 38.18 * N + 4.81


def Y(r, theta):
    return 1


def sif(sigma, r, theta):
    return Y(r, theta) * sigma * np.sqrt(np.pi * r)


# -------------- Propagation Models


def propagation_model(model: str, r: float, theta: float, load: list):
    sif_max = sif(load[0], r, theta)
    sif_min = sif(load[1], r, theta)
    K = sif_max - sif_min

    if model == 'Paris':
        return C * pow(K, n) + r
    if model == 'Nasgrow':
        pass
    if model == 'Walker':
        pass
    else:
        print('Model not available or mispelled')


# --------------

class CrackPropagator:

    def __init__(self, init_size: float = 1, file: str = 'sequence.csv', model: str = 'Paris',
                 nr_dots: int = 6):
        self.init_size = init_size
        self.file = file
        self.model = model
        self.nr_dots = nr_dots
        self.thetas = np.linspace(0, math.pi, nr_dots)
        self.crack_size = [self.get_radius(t) for t in self.thetas]
        self.info()

        # load flight history
        self.df = pd.read_csv(self.file)

        self.history = []
        self.iterate()

    def get_radius(self, theta):
        return self.init_size / (math.sqrt(math.cos(theta) ** 2 + 4 * math.sin(theta) ** 2))

    def iterate(self):
        load_sequence = self.df.load.tolist()
        load_matrix = to_matrix(load_sequence, 2)

        for l in load_matrix:
            increments = [propagation_model(self.model, r, self.thetas[i], l) for i, r in
                          enumerate(self.crack_size)]
            self.history.append(increments)
            self.crack_size[:] = increments
        print(self.history)

    def info(self):
        attrs = vars(self)
        print('# Crack Propagation Class -  Information\n')
        print(', \n'.join("%s: %s" % item for item in attrs.items()))


# Creating dot class
class Dot(object):
    def __init__(self, radius, theta):
        self.radius = radius
        self.theta = theta
        # print(f'Created Dot with radius {self.radius} and theta {self.theta}')

    def move(self, radius):
        self.radius = radius


def plot(C: CrackPropagator):
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    ax.set_ylim(0, 15)

    dots = [Dot(C.get_radius(C.thetas[i]), C.thetas[i]) for i in range(C.nr_dots)]
    d, = ax.plot([dot.theta for dot in dots],
                  [dot.radius for dot in dots], '-ro')
    the_plot = st.pyplot(fig, clear_figure=False)

    def init():
        dots = [Dot(C.get_radius(C.thetas[i]), C.thetas[i]) for i in range(C.nr_dots)]
        d.set_data([dot.theta for dot in dots],
                   [dot.radius for dot in dots])
        return d,

    def animate(i):
        for p, dot in enumerate(dots):
            dot.move(C.history[i][p])
        d.set_data([dot.theta for dot in dots],
                   [dot.radius for dot in dots])
        the_plot.pyplot(fig, clear_figure=False)

    if st.button(" Click for Animation"):
        for f in range(3):
            init()
            for i in range(6):
                animate(i)
                time.sleep(0.1)


# Press the green button in the gutter to run the script.
def main():
    default_init = 1.
    default_nr_dots = 6

    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.header("Crack Propagator.")
    st.write("Please configure inputs on the left.")

    # Input Selection
    init_size = st.sidebar.number_input("Enter initial crack length", 0., 10., default_init)  # Initial Crack Size

    nr_dots = st.sidebar.number_input("Enter Number of Dots", 2, 20, default_nr_dots)  # Number of Dots

    model = st.sidebar.selectbox('Select Propagation Model',  # Propagation Law
                         ('Paris', 'Nasgrow', 'Walker'))
  
    st.sidebar.write('You selected:', model)

    if uploaded_files is None:
        uploaded_files = default_file
        st.write('Default File Uploaded Successfuly!')
    else:
        st.write('Your File was Uploaded Successfuly!')

    C = CrackPropagator(init_size=init_size, model=model, file=uploaded_files, nr_dots=nr_dots)
    visualize_data(C.df, C.df.columns[0], C.df.columns[1])

    plot(C)

    if st.button('Get Crack Final Size'):
        st.write('Final Crack Lenghts ', C.crack_size)


def visualize_data(df, x_axis, y_axis):
    graph = alt.Chart(df).mark_line().encode(
        x=x_axis,
        y=y_axis
    ).interactive()

    st.write(graph)


def to_matrix(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]


if __name__ == '__main__':
    main()
