import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from matplotlib.patches import Patch

def format_score(val):
    """
    Formats numerical values for display in plots.
    
    Args:
        val (float): The numerical value to be formatted.
    
    Returns:
        str: Formatted string representation of the value.
    """
    return f"{val:.5f}" if val >= 0.001 else "<0.001"


def plot_bar(data, 
             x = None, 
             y=None, 
             barmode="group", 
             text="formatted_score", 
             title = None, 
             facet_col=None, 
             height=400,
             width = 1300,
             figure_path = None
            ):
    """
    Generates a bar plot using Plotly.
    
    Args:
        data (pd.DataFrame): Dataframe containing the data to be plotted.
        x (str, optional): Column name for x-axis values.
        y (str, optional): Column name for y-axis values.
        barmode (str, optional): Mode for bars ('group' or 'stack'). Defaults to 'group'.
        text (str, optional): Column to be displayed as text labels on bars.
        title (str, optional): Title of the plot.
        facet_col (str, optional): Column used for creating faceted subplots.
        height (int, optional): Height of the plot in pixels.
        width (int, optional): Width of the plot in pixels.
        figure_path (str, optional): Path to save the plot.
    
    Returns:
        None. Displays the bar plot and saves the image if a path is provided.
    """
    fig = px.bar(data, y=y, x=x, barmode=barmode, text=text, title=title, facet_col=facet_col)
    fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    
    # Customize layout
    fig.update_layout(
        height=height,  
        width=width,
        title_x=0.5,  # Centers the title
        title_font=dict(size=20),
        font=dict(size=14)
    )

    fig.write_image(figure_path, scale=6)
    

def plot_heatmap(df, 
                 figure_path, 
                 filter:bool = False, 
                 standard_scale = None, 
                 legend_label:str = 'Scaled Diff Scores', 
                 ylabel_cutoff:int = 75, 
                 cluster_rows:bool = False, 
                 cluster_columns = False,
                 annotations = None,
                 linewidths=0.3, 
                 linecolor="gray",
                 figsize = (12,6)
                ):
    """
    Generates a heatmap visualization with optional clustering and group color annotations.
    
    Args:
        df (pd.DataFrame): Dataframe containing the values to be plotted.
        figure_path (str): Path where the heatmap image will be saved.
        filter (bool, optional): If True, filters top pathways based on mean absolute difference.
        standard_scale (int, optional): If set, standardizes values before plotting.
        legend_label (str, optional): Label for the color bar.
        ylabel_cutoff (int, optional): Character cutoff for y-axis labels.
        cluster_rows (bool, optional): If True, clusters rows.
        cluster_columns (bool, optional): If True, clusters columns.
        group_labels (list, optional): Labels for grouping rows with corresponding colors.
    
    Returns:
        None. Saves the heatmap plot as an image.
    """
    df = df.copy()
    df.index = [name[:ylabel_cutoff] for name in df.index]
    
    if filter:
        # Compute pathway difference means
        means = df.mean(axis=1).abs()
    
        # Retain top pathways based on mean absolute difference
        cutoff = sorted(means, reverse=True)[min(40, df.shape[0]-1)]
        df = df[means > cutoff]

    if annotations is None:
        annotations = {
            "row": {
                "row_colors": None,
                "group_palette": None
            },
            "column": {
                "col_colors": None,
                "group_palette": None
            }
        }
    row_colors = annotations["row"]["row_colors"]
    col_colors = annotations["column"]["col_colors"]

    clustermap = sns.clustermap(df, 
                                cmap="RdBu_r", 
                                center=0, 
                                cbar_kws={'label': legend_label},
                                figsize=figsize, 
                                row_cluster=cluster_rows, 
                                col_cluster=cluster_columns, 
                                row_colors=row_colors,
                                col_colors=col_colors,
                                standard_scale= standard_scale,
                                linewidths=linewidths, 
                                linecolor=linecolor
                               )
    
    # Rotate x-axis labels
    for label in clustermap.ax_heatmap.get_xticklabels():
        label.set_rotation(45)  

    # Keep y-axis labels horizontal
    for text in clustermap.ax_heatmap.get_yticklabels():
        text.set_rotation(0)

    # Create Annotation Legends
    if annotations["column"]["group_palette"] is not None:
        # Create a legend for column colors
        legend_patches = [Patch(color=color, label=group) for group, color in annotations["column"]["group_palette"].items()]
        plt.legend(handles=legend_patches, title="Group", bbox_to_anchor=(1.7, 0.25), loc="lower left", borderaxespad=0)

        
    # plt.tight_layout()
    plt.savefig(figure_path, dpi=600, bbox_inches='tight')
    # plt.show()
    plt.close()
