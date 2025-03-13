import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

COLORS = np.array([
  [  0,   0,   0],  # unlabeled    =   0,
  [ 70,  70,  70],  # building     =   1,
  [190, 153, 153],  # fence        =   2, 
  [250, 170, 160],  # other        =   3,
  [220,  20,  60],  # pedestrian   =   4, 
  [153, 153, 153],  # pole         =   5,
  [157, 234,  50],  # road line    =   6, 
  [128,  64, 128],  # road         =   7,
  [244,  35, 232],  # sidewalk     =   8,
  [107, 142,  35],  # vegetation   =   9, 
  [  0,   0, 142],  # car          =  10,
  [102, 102, 156],  # wall         =  11, 
  [220, 220,   0],  # traffic sign =  12,
  [ 60, 250, 240],  # anomaly      =  13,
]) 

def color(img_np:np.ndarray, colors:np.ndarray) -> Image.Image:
    """
    Source: https://github.com/hendrycks/anomaly-seg/issues/15#issuecomment-890300278
    """
    img_new = np.zeros((img_np.shape[0],img_np.shape[1],3))

    for index, color in enumerate(colors):
      img_new[img_np == index] = color
    
    return Image.fromarray(img_new.astype("uint8"), "RGB")


def plot_results(df, query: str, sort_col: str, log_scale: bool = False, figsize=(8, 4)):
    """
    Generates two graphs from a pandas dataframe based on the specified
    query and parameters.

    Parameters:
    df (DataFrame): The dataframe containing the data.
    query (str): The query string to filter the dataframe.
    sort_col (str): Column for sorting.
    log_scale (bool): Use a symmetric log scale for the X-axis.
    """
    subset = df.query(query).sort_values(by=sort_col, ascending=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.plot(subset[sort_col].to_numpy(), subset['mIoU'].to_numpy(), '-o', label='mIoU')
    ax1.set_xlabel(sort_col.replace('_', ' ').title())
    ax1.set_ylabel("mIoU")
    ax1.tick_params(axis='y')
    ax1.grid(True, which="major", linewidth=0.5)

    if log_scale:
        ax1.set_xscale('symlog', linthresh=1e-2)
        ax2.set_xscale('symlog', linthresh=1e-2)
    else:
        ax1.set_xticks(range(int(subset[sort_col].min()), int(subset[sort_col].max()) + 1))
        ax2.set_xticks(range(int(subset[sort_col].min()), int(subset[sort_col].max()) + 1))

    ax2.plot(subset[sort_col].to_numpy(), subset['entropy'].to_numpy(), '-o', color='r', label='Entropy')
    ax2.plot(subset[sort_col].to_numpy(), subset['msp'].to_numpy(), '-o', label='Msp')
    ax2.plot(subset[sort_col].to_numpy(), subset['maxlog'].to_numpy(), '-o', label='MaxLog')
    ax2.plot(subset[sort_col].to_numpy(), subset['energy'].to_numpy(), '-o', label='Energy')
    ax2.set_xlabel(sort_col.replace('_', ' ').title())
    ax2.set_ylabel("AUPR")
    ax2.tick_params(axis='y')
    ax2.grid(True, which='major', linewidth=0.5)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=4, frameon=False, fontsize='small')

    fig.tight_layout()
    plt.show()


def draw_barplot(df, metric:str, name:str, comparison_params:list = None,
                 query_group1:str = None, query_group2:str = None,
                 label_group1:str = None, label_group2:str = None,
                 figsize=(10, 3)):
    """
    Draw a barplot comparing two groups of a pandas dataframe.

    Parameters:
      - df (DataFrame): The dataframe containing the data.
      - metric (str): Metric to be used.
      - name (str): Label for the y-axis.
      - comparison_params (list): List of columns for ordering and creating ticks.
      - query_group1 (str): Query string for the first group.
      - query_group2 (str): Query string for the second group.
      - label_group1 (str): Label for the first group.
      - label_group2 (str): Label for the second group.
      - figsize (tuple): Size of the figure.
    """
    sorted_df = df.sort_values(comparison_params)
    group1_df = sorted_df.query(query_group1)
    group2_df = sorted_df.query(query_group2)

    group1_scores = group1_df[metric].to_numpy()
    group2_scores = group2_df[metric].to_numpy()

    if comparison_params is not None:
        ticks_df = group1_df[comparison_params].astype(str)
        ticks = ticks_df.apply(lambda row: "\n".join(row), axis=1).tolist()
        ticks = ['\n'.join(comparison_params).title()] + ticks
    else:
        ticks = group1_df.index.astype(str).tolist()

    num_bars = len(group1_scores)

    bar_positions = np.arange(num_bars) + 1.2
    tick_positions = np.arange(num_bars + 1) + 0.2
    tick_positions[0] = 0.0
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    bars1 = ax.bar(bar_positions - width / 2, group1_scores, width, label=label_group1)
    bars2 = ax.bar(bar_positions + width / 2, group2_scores, width, label=label_group2)

    ylim_lower = min(group1_scores.min(), group2_scores.min()) - 1
    ylim_upper = max(group1_scores.max(), group2_scores.max()) + 2
    ax.set(
        ylabel=name,
        xticks=tick_positions,
        xticklabels=ticks,
        ylim=[ylim_lower, ylim_upper]
    )

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), frameon=False, ncol=2)

    for group in [bars1, bars2]:
        for bar in group:
            ax.annotate(
                f'{bar.get_height():.2f}',
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3), textcoords='offset points',
                ha='center', va='bottom', fontsize=9
            )

    for tick in ax.get_xticklabels():
        tick.set_color('red')
        tick.set_horizontalalignment('left')
        break

    plt.tight_layout()
    plt.show()


def draw_loss_barplot(df, query:str, figsize=(10,3)):
  """
  Draw a barplot comparing losses over all the metrics on a pandas dataframe.

  Parameters:
    - df (DataFrame): The dataframe containing the data.
    - query (str): Query string.
    - figsize (tuple): Size of the figure.
  """
  names = ['msp AUPR', 'maxlogit AUPR', 'entropy AUPR', 'energy AUPR']
  metrics = ['msp','maxlog','entropy','energy']

  ce_scores  = df.query(query+" and loss=='CE'")[metrics].to_numpy()
  fl_scores  = df.query(query+" and loss=='FL'")[metrics].to_numpy()
  ce_h_scores  = df.query(query+" and loss=='CE+H'")[metrics].to_numpy()
  fl_h_scores  = df.query(query+" and loss=='FL+H'")[metrics].to_numpy()

  x = (4*0.35+0.35)*np.arange(4)
  width = 0.35

  fig, ax = plt.subplots(figsize=figsize)
  bars = [ax.bar(x-3*width/2, ce_scores[0], width, label='CE'),
          ax.bar(x-width/2, fl_scores[0], width, label='FL'),
          ax.bar(x+width/2, ce_h_scores[0], width, label='CE+H'),
          ax.bar(x+3*width/2, fl_h_scores[0], width, label='FL+H')]

  ax.set(xticks=x, xticklabels=names, ylim=[0,20])
  ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=4, frameon=False)

  for group in bars:
    for bar in group:
      ax.annotate(f'{bar.get_height():.2f}',
                  (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                  xytext=(0, 3), textcoords='offset points',
                  ha='center', va='bottom')
  plt.tight_layout()
  plt.show()