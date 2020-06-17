import pickle
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import time


def create_data_on_disk(graph_size, num_samples, is_save=True, filename=None, is_return=False, seed=1234):
    """Generate validation dataset (with SEED) and save
    """

    CAPACITIES = {
        10: 20.,
        20: 30.,
        50: 40.,
        100: 50.
    }
    depo, graphs, demand = (tf.random.uniform(minval=0, maxval=1, shape=(num_samples, 2), seed=seed),
                            tf.random.uniform(minval=0, maxval=1, shape=(num_samples, graph_size, 2), seed=seed),
                            tf.cast(tf.random.uniform(minval=1, maxval=10, shape=(num_samples, graph_size),
                                                      dtype=tf.int32, seed=seed), tf.float32) / tf.cast(CAPACITIES[graph_size], tf.float32)
                            )
    if is_save:
        save_to_pickle('Validation_dataset_{}.pkl'.format(filename), (depo, graphs, demand))

    if is_return:
        return tf.data.Dataset.from_tensor_slices((list(depo), list(graphs), list(demand)))


def save_to_pickle(filename, item):
    """Save to pickle
    """
    with open(filename, 'wb') as handle:
        pickle.dump(item, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_from_pickle(path, return_tf_data_set=True, num_samples=None):
    """Read dataset from file (pickle)
    """

    objects = []
    with (open(path, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    objects = objects[0]
    if return_tf_data_set:
        depo, graphs, demand = objects
        if num_samples is not None:
            return tf.data.Dataset.from_tensor_slices((list(depo), list(graphs), list(demand))).take(num_samples)
        else:
            return tf.data.Dataset.from_tensor_slices((list(depo), list(graphs), list(demand)))
    else:
        return objects


def generate_data_onfly(num_samples=10000, graph_size=20):
    """Generate temp dataset in memory
    """

    CAPACITIES = {
        10: 20.,
        20: 30.,
        50: 40.,
        100: 50.
    }
    depo, graphs, demand = (tf.random.uniform(minval=0, maxval=1, shape=(num_samples, 2)),
                            tf.random.uniform(minval=0, maxval=1, shape=(num_samples, graph_size, 2)),
                            tf.cast(tf.random.uniform(minval=1, maxval=10, shape=(num_samples, graph_size),
                                                      dtype=tf.int32), tf.float32)/tf.cast(CAPACITIES[graph_size], tf.float32)
                            )

    return tf.data.Dataset.from_tensor_slices((list(depo), list(graphs), list(demand)))


def get_results(train_loss_results, train_cost_results, val_cost, save_results=True, filename=None, plots=True):

    epochs_num = len(train_loss_results)

    df_train = pd.DataFrame(data={'epochs': list(range(epochs_num)),
                                  'loss': train_loss_results,
                                  'cost': train_cost_results,
                                  })
    df_test = pd.DataFrame(data={'epochs': list(range(epochs_num)),
                                 'val_сost': val_cost})
    if save_results:
        df_train.to_excel('train_results_{}.xlsx'.format(filename), index=False)
        df_test.to_excel('test_results_{}.xlsx'.format(filename), index=False)

    if plots:
        plt.figure(figsize=(15, 9))
        ax = sns.lineplot(x='epochs', y='loss', data=df_train, color='salmon', label='train loss')
        ax2 = ax.twinx()
        sns.lineplot(x='epochs', y='cost', data=df_train, color='cornflowerblue', label='train cost', ax=ax2)
        sns.lineplot(x='epochs', y='val_сost', data=df_test, palette='darkblue', label='val cost').set(ylabel='cost')
        ax.legend(loc=(0.75, 0.90), ncol=1)
        ax2.legend(loc=(0.75, 0.95), ncol=2)
        ax.grid(axis='x')
        ax2.grid(True)
        plt.savefig('learning_curve_plot_{}.jpg'.format(filename))
        plt.show()


def get_journey(batch, pi, title, ind_in_batch=0):
    """Plots journey of agent

    Args:
        batch: dataset of graphs
        pi: paths of agent obtained from model
        ind_in_batch: index of graph in batch to be plotted
    """

    # Remove extra zeros
    pi_ = get_clean_path(pi[ind_in_batch].numpy())

    # Unpack variables
    depo_coord = batch[0][ind_in_batch].numpy()
    points_coords = batch[1][ind_in_batch].numpy()
    demands = batch[2][ind_in_batch].numpy()
    node_labels = ['(' + str(x[0]) + ', ' + x[1] + ')' for x in enumerate(demands.round(2).astype(str))]

    # Concatenate depot and points
    full_coords = np.concatenate((depo_coord.reshape(1, 2), points_coords))

    # Get list with agent loops in path
    list_of_paths = []
    cur_path = []
    for idx, node in enumerate(pi_):

        cur_path.append(node)

        if idx != 0 and node == 0:
            if cur_path[0] != 0:
                cur_path.insert(0, 0)
            list_of_paths.append(cur_path)
            cur_path = []

    list_of_path_traces = []
    for path_counter, path in enumerate(list_of_paths):
        coords = full_coords[[int(x) for x in path]]

        # Calculate length of each agent loop
        lengths = np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1))
        total_length = np.sum(lengths)

        list_of_path_traces.append(go.Scatter(x=coords[:, 0],
                                              y=coords[:, 1],
                                              mode="markers+lines",
                                              name=f"path_{path_counter}, length={total_length:.2f}",
                                              opacity=1.0))

    trace_points = go.Scatter(x=points_coords[:, 0],
                              y=points_coords[:, 1],
                              mode='markers+text',
                              name='destinations',
                              text=node_labels,
                              textposition='top center',
                              marker=dict(size=7),
                              opacity=1.0
                              )

    trace_depo = go.Scatter(x=[depo_coord[0]],
                            y=[depo_coord[1]],
                            text=['1.0'], textposition='bottom center',
                            mode='markers+text',
                            marker=dict(size=15),
                            name='depot'
                            )

    layout = go.Layout(title='<b>Example: {}</b>'.format(title),
                       xaxis=dict(title='X coordinate'),
                       yaxis=dict(title='Y coordinate'),
                       showlegend=True,
                       width=1000,
                       height=1000,
                       template="plotly_white"
                       )

    data = [trace_points, trace_depo] + list_of_path_traces
    print('Current path: ', pi_)
    fig = go.Figure(data=data, layout=layout)
    fig.show()

def get_cur_time():
    """Returns local time as string
    """
    ts = time.time()
    return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')


def get_clean_path(arr):
    """Returns extra zeros from path.
       Dynamical model generates duplicated zeros for several graphs when obtaining partial solutions.
    """

    p1, p2 = 0, 1
    output = []

    while p2 < len(arr):

        if arr[p1] != arr[p2]:
            output.append(arr[p1])
            if p2 == len(arr) - 1:
                output.append(arr[p2])

        p1 += 1
        p2 += 1

    if output[0] != 0:
        output.insert(0, 0.0)
    if output[-1] != 0:
        output.append(0.0)

    return output

