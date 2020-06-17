import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def f_get_results_plot_seaborn(data, title, graph_size=20):
    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_subplot()
    ax.plot(data['epochs'], data['train_loss'], color='salmon', label='train loss')
    ax2 = ax.twinx()
    ax2.plot(data['epochs'], data['train_cost'],  color='cornflowerblue', label='train cost')
    ax2.plot(data['epochs'], data['val_cost'], color='darkblue', label='val cost')

    if graph_size == 20:
        am_val = 6.4
    else:
        am_val = 10.98

    plt.axhline(y=am_val, color='black', linestyle='--', linewidth=1.5, label='AM article best score')

    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
    
    ax.set_ylabel('Loss')
    ax2.set_ylabel('Cost')
    ax.set_xlabel('Epochs')
    ax.grid(False)
    ax2.grid(False)
    ax2.set_yticks(np.arange(min(data['val_cost'].min(), data['train_cost'].min())-0.2,
                             max(data['val_cost'].max(), data['train_cost'].max())+0.1,
                             0.1).round(2))
    plt.title('Learning Curve: ' + title)
    plt.show()


def f_get_results_plot_plotly(data, title, graph_size=20):
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Scatter(x=data['epochs'], y=data['train_loss'], name="train loss", marker_color='salmon'),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=data['epochs'], y=data['train_cost'], name="train cost", marker_color='cornflowerblue'),
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(x=data['epochs'], y=data['val_cost'], name="val cost", marker_color='darkblue'),
        secondary_y=True,
    )

    # Add figure title
    fig.update_layout(
        title_text="Learning Curve: " + title,
        width=950,
        height=650,
        # plot_bgcolor='rgba(0,0,0,0)'
        template="plotly_white"
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Number of epoch")

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Loss", secondary_y=False, showgrid=False, zeroline=False)
    fig.update_yaxes(title_text="<b>Cost", secondary_y=True, dtick=0.1)#, nticks=20)

    fig.show()