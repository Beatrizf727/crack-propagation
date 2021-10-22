import pandas as pd
import streamlit as st
import altair as alt
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import math
import time
import pickle
from tqdm import tqdm

# -------------- Constants

# Paris
n_paris = 2.7026
C_paris = 1.363 * 10 ** (-10)  # / 10 ** (-3-3*n_paris/2)

# Nasgro

C_nasgro = 7.23 * 10 ** (-11)
n_nasgro = 3.6
q = 1
p = 0.5
delta_Kth = 0.879
alpha = 2.
s_max_sigma_0 = 0.3
A_0 = (0.825 - 0.34 * alpha + 0.05 * alpha ** 2) * ((np.cos(np.pi * s_max_sigma_0 / 2.)) ** (1 / alpha))
A_1 = (0.415 - 0.071 * alpha) * s_max_sigma_0
A_3 = 2 * A_0 + A_1 - 1
A_2 = 1 - A_0 - A_1 - A_3


def f(R: float):
    result = 0
    if R >= 0:
        result = max(R, A_0 + A_1 * R + A_2 * R ** 2 + A_3 * R ** 3)
    elif -2 <= R < 0:
        result = A_0 + A_1 * R
    elif R < -2:
        result = A_0 - 2 * A_1
    return result


# Walker Constants

C_walker = 1.1824 * 10 ** (-11)
n_walker = 3.815
gamma = 0.691

p00 = 0.4306
p10 = 0.1175
p01 = 0.1437
p20 = -0.09335
p11 = -0.04874
p02 = 0.03878
p30 = 0.03478
p21 = 0.01998
p12 = 0.01074
p03 = -0.05319
p40 = -0.005875
p31 = -0.005317
p22 = -0.0001824
p13 = -0.002181
p04 = 0.0086
p50 = 0.0003921
p41 = 0.0004211
p32 = 1.646e-05
p23 = 0.0003424
p14 = 0.0001549

# -------------- Functions

def load_to_tension(N: float):
    return 33.7 * N


def Y(a, theta):
    x = a
    y = theta

    return (
            p00 + p10 * x + p01 * y + p20 * x ** 2 + p11 * x * y + p02 * y ** 2 + p30 * x ** 3 + p21 * x ** 2 * y
            + p12 * x * y ** 2 + p03 * y ** 3 + p40 * x ** 4 + p31 * x ** 3 * y + p22 * x ** 2 * y ** 2
            + p13 * x * y ** 3 + p04 * y ** 4 + p50 * x ** 5 + p41 * x ** 4 * y + p32 * x ** 3 * y ** 2
            + p23 * x ** 2 * y ** 3 + p14 * x * y ** 4
    )


def sif(sigma, a, theta):
    return Y(a, theta) * sigma * np.sqrt(np.pi * a * 10 ** (-3))  # convert to MPa/sqrt(m)


# -------------- Propagation Models


def propagation_model(model: str, r: float, a: float, theta: float, load: list, prints: bool = False):
    if load[0] >= load[1]:
        print('Valley greater than peak encountered')
        print(load[0], load[1])
        exit()

    sigma_1 = load_to_tension(load[0])
    sigma_2 = load_to_tension(load[1])

    sif_min = sif(sigma_1, a, theta)
    sif_max = sif(sigma_2, a, theta)

    R = sif_min / sif_max
    delta_K = sif_max - sif_min

    if delta_K > 84:
        print('Critical Limit Surpassed')
        return (None, None)

    if prints:
        print('loads', load[0], load[1])
        print('sifs', sif_max, sif_min)
        print('R: ', R)
        print('delta_K ', delta_K)

    if model == 'Paris':
        return r + C_paris * pow(delta_K, n_paris) * 10 ** 3

    if model == 'Nasgro':
        if delta_K >= 0.879:  # non propagation limit
            r_next = r + C_nasgro * 10 ** 3 * (delta_K * (1 - f(R)) / (1 - R)) ** n_nasgro * (
                        1 - (delta_Kth / delta_K)) ** p / (
                             1 - (sif_max / 84.)) ** q
        else:
            # Automatic Increment
            r_next = r * 1
        return (r_next, delta_K)

    if model == 'Walker':
        if delta_K >= 2.835:
            r_next = r + (C_walker * (delta_K / ((1 - R) ** (1 - gamma))) ** n_walker) * 10 ** 3
        else:
            # Automatic Increment
            r_next = r * 1

        if np.isnan(r_next):
            print(r_next)
            print('increment is nan')
            exit()
        return (r_next, delta_K)

    else:
        print('Model not available or mispelled')


# --------------

class CrackPropagator:

    def __init__(self, init_size: float = 1, file: str = 'sequence.csv', repetitions: int = 1,
                 model: str = 'Paris', nr_dots: int = 6, save_simulation: bool = False):
        self.init_size = init_size
        self.file = file
        self.model = model
        self.nr_dots = nr_dots
        self.thetas = np.linspace(0, math.pi, nr_dots)
        self.crack_size = [self.get_radius(t) for t in self.thetas]
        self.info()

        # load flight history
        self.df = pd.read_csv(self.file, sep=',')
        self.df = pd.concat([self.df] * repetitions, ignore_index=True)
        print(self.df.head())

        self.history = []
        self.crack_over_time = []
        self.deltaK_over_time = []
        self.iterate()

        if save_simulation:
            with open('simulation_files/' + file.split('/')[-1].split('.')[0] + '_' + model + '_' + str(init_size) +
                      '_' + str(repetitions) + '.pkl', 'wb') as f:
                pickle.dump(self.crack_over_time, f)
                pickle.dump(self.deltaK_over_time, f)

    def get_radius(self, theta):
        return self.init_size / (math.sqrt(math.cos(theta) ** 2 + 4 * math.sin(theta) ** 2))

    def iterate(self):
        load_sequence = self.df['load'].tolist()
        load_matrix = to_matrix(load_sequence, 2)

        a = self.init_size / 2.
        for i, l in enumerate(tqdm(load_matrix)):
            increments = [propagation_model(self.model, r, a, self.thetas[i], l)[0] for i, r in
                          enumerate(self.crack_size)]
            delta_K = [propagation_model(self.model, r, a, self.thetas[i], l)[1] for i, r in
                          enumerate(self.crack_size)]

            # self.history.append(increments)
            if None in increments:
                print('stopping criteria found')
                break

            if i % 5000 == 0:
                self.crack_over_time.append(increments[1])
                self.deltaK_over_time.append(delta_K[1])
            self.history.append(increments)
            self.crack_size = increments
            a = increments[int(self.nr_dots / 2)]
        #print(self.history)

    def info(self):
        attrs = vars(self)
        print('# Crack Propagation Class -  Information\n')
        print(', \n'.join("%s: %s" % item for item in attrs.items()))

    def plot_over_time(self):
        plt.plot(self.crack_over_time, '-ro')
        plt.show()


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
    ax.set_ylim(0, 2.5)

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
            for i in range(1, 500, 100):
                animate(i)
                time.sleep(0.1)


def visualize_data(df, x_axis, y_axis):
    graph = alt.Chart(df).mark_line().encode(
        x=x_axis,
        y=y_axis
    ).interactive()
    graph.encoding.x.title = 'Time (s)'
    graph.encoding.y.title = 'Load Factor'
    graph.encoding.y.scale = alt.Scale(domain=[-1, 6])
    alt.data_transformers.disable_max_rows()

    #graph.show()
    st.write(graph)


def to_matrix(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]


# Press the green button in the gutter to run the script.
def main():
    default_init = 2.
    default_nr_dots = 6
    default_file = 'flight_data/acrobacias/sequence.csv'
    #default_file = 'flight_data/acrobacias/acrobacias_8.csv'

    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.header("Crack Propagator.")
    st.write("Please configure inputs on the left.")

    # Input Selection
    init_size = st.sidebar.number_input("Enter initial crack length (mm)", 1., 4.,
                                        default_init)  # *10**(-3)  # Initial Crack Size

    nr_dots = st.sidebar.number_input("Enter Number of Dots", 2, 20, default_nr_dots)  # Number of Dots

    model = st.sidebar.selectbox('Select Propagation Model',  # Propagation Law
                                 ('Nasgro', 'Walker'))

    st.sidebar.write('You selected:', model)

    uploaded_files = st.sidebar.file_uploader("Choose CSV file", accept_multiple_files=False)

    if uploaded_files is None:
        uploaded_files = default_file
        st.write('Default File Uploaded Successfuly!')
    else:
        st.write('Your File was Uploaded Successfuly!')

    C = CrackPropagator(init_size=init_size, model=model, file=uploaded_files, nr_dots=nr_dots, repetitions=1)
    visualize_data(C.df, C.df.columns[0], C.df.columns[1])

    plot(C)

    if st.button('Get Crack Final Size'):
        st.write('Final Crack Lenghts ', C.crack_size)


def simulate():
    """

    C4 = CrackPropagator(init_size=1, model='Paris', file='flight_data/instrumentos/instrumentos_0.csv',
                        nr_dots=3, repetitions=5000, save_simulation=True)


    C1 = CrackPropagator(init_size=1.5, model='Paris', file='flight_data/acrobacias/acrobacias_9.csv',
                         nr_dots=3, repetitions=20000, save_simulation=True)

    C1 = CrackPropagator(init_size=1, model='Nasgro', file='flight_data/acrobacias/acrobacias_9.csv',
                         nr_dots=3, repetitions=20000, save_simulation=True)


    C3 = CrackPropagator(init_size=1, model='Walker', file='flight_data/acrobacias/acrobacias_9.csv',
                         nr_dots=3, repetitions=15000, save_simulation=True)


    C6 = CrackPropagator(init_size=1, model='Walker', file='flight_data/instrumentos/instrumentos_0.csv',
                         nr_dots=3, repetitions=15000, save_simulation=True)


    #C5 = CrackPropagator(init_size=1, model='Nasgro', file='flight_data/instrumentos/instrumentos_0.csv',
    #                    nr_dots=5, repetitions=15000, save_simulation=True)

    #visualize_data(C.df, C.df.columns[0], C.df.columns[1]) # uncomment to view spectrum
    #print('Final Crack Lenghts ', C.crack_size)
     #C.plot_over_time()

     """
    # Acrobacias
    #C1 = CrackPropagator(init_size=2, model='Nasgro', file='flight_data/acrobacias/acrobacias_8.csv',
    #nr_dots=3, repetitions=20000, save_simulation=True)
   # print('Final Crack Lenghts ', C1.crack_size)

    #C2 = CrackPropagator(init_size=2, model='Walker', file='flight_data/acrobacias/acrobacias_8.csv',
                     #   nr_dots=3, repetitions=20000, save_simulation=True)
    #print('Final Crack Lenghts ', C2.crack_size)
    #visualize_data(C2.df, C2.df.columns[0], C2.df.columns[1])

    # Instrumentos


    #C4 = CrackPropagator(init_size=2, model='Nasgro', file='flight_data/instrumentos/instrumentos_0.csv',
    #                     nr_dots=3, repetitions=40000, save_simulation=True)
    #print('Final Crack Lenghts ', C4.crack_size)

    C3 = CrackPropagator(init_size=2, model='Walker', file='flight_data/instrumentos/instrumentos_0.csv',
                         nr_dots=3, repetitions=1, save_simulation=False)
    print('Final Crack Lenghts ', C3.crack_size)

def create_bigfile(flight_type: str = 'acrobacias'):
    if flight_type == 'acrobacias':
        #df = pd.read_csv('flight_data/instrumentos/normal_3.csv', sep=',')
        #df = pd.read_csv('flight_data/instrumentos/normal_2.csv', sep=',')
        #df = pd.read_csv('flight_data/instrumentos/normal_3.csv', sep=',')
        #df = pd.read_csv('flight_data/acrobacias/acrobacias_1.csv', sep=',')
        #df = pd.read_csv('flight_data/acrobacias/acrobacias_2.csv', sep=',')
        #df = pd.read_csv('flight_data/acrobacias/acrobacias_3.csv', sep=',')
        #df2 = pd.read_csv('flight_data/acrobacias/acrobacias_5.csv', sep=',')
        #df1 = pd.read_csv('flight_data/acrobacias/acrobacias_5.csv', sep=',')


        #df = pd.concat([df, df1], ignore_index=True).reindex(df.index)
        #df = pd.concat([df, df2], ignore_index=True).reindex(df.index)
        #df = pd.concat([df, df3], ignore_index=True).reindex(df.index)
        #df = pd.concat([df, df4], ignore_index=True).reindex(df.index)
        #df = pd.concat([df, df5], ignore_index=True).reindex(df.index)
        #df = pd.concat([df, df6], ignore_index=True).reindex(df.index)
        #df = pd.concat([df, df7], ignore_index=True).reindex(df.index)

        df.to_csv('flight_data/acrobacias/acrobacias_0.csv', index=False)

    elif flight_type == 'instrumentos':

        df = pd.read_csv('flight_data/acrobacias/acrobacias_3.csv', sep=',')
        #df2 = pd.read_csv('flight_data/instrumentos/normal_2.csv', sep=',')
        df1 = pd.read_csv('flight_data/instrumentos/normal_3.csv', sep=',')
        #df2 = pd.read_csv('flight_data/acrobacias/acrobacias_5.csv', sep=',')
        #df3 = pd.read_csv('flight_data/instrumentos/normal_4.csv', sep=',')
        #df4 = pd.read_csv('flight_data/acrobacias/acrobacias_1.csv', sep=',')
        #df5 = pd.read_csv('flight_data/acrobacias/acrobacias_2.csv', sep=',')

        df = pd.concat([df, df1], ignore_index=True).reindex(df.index)
        #df = pd.concat([df, df2], ignore_index=True).reindex(df.index)
        #df = pd.concat([df, df4], ignore_index=True).reindex(df.index)
        #df = pd.concat([df, df5], ignore_index=True).reindex(df.index)
        #df = pd.concat([df, df3], ignore_index=True).reindex(df.index)
        #df = pd.concat([df, df7], ignore_index=True).reindex(df.index)

        df.to_csv('flight_data/instrumentos/instrumentos_0.csv', index=False)
        print('instrumentos file created')

   # elif flight_type == 'other':
        #df = pd.read_csv('flight_data/instrumentos/instrumentos_0.csv', sep=',')
        # df1 = pd.read_csv('flight_data/acrobacias/acrobacias_2.csv', sep=',')

        # df = pd.concat([df, df1], ignore_index=True).reindex(df.index)

        #df.to_csv('flight_data/instrumentos/instrumentos_10.csv', index=False)


if __name__ == '__main__':
    main()
    #create_bigfile(flight_type='acrobacias')
    #create_bigfile(flight_type='instrumentos')
    #simulate()
