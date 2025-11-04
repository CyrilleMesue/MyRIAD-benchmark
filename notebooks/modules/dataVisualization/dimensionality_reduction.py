# Load Packages
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly
from matplotlib.pyplot import figure
from mpl_toolkits.mplot3d import Axes3D
sns.set_theme(rc={'figure.figsize':(12,8)})

# custom modules
from modules.logger import logging
from modules.exception import CustomException

class Dimensionality_Reduction:
    """
    A class to perform and visualize dimensionality reduction using PCA, t-SNE, and UMAP 
    in both 2D and 3D scatter plots. The plots can be generated using either Seaborn or Plotly.
    """

    def __init__(self, 
                 data= None, 
                 labels_column:str = "Diagnosis", 
                 dim:str = "2D", 
                 method:str = "pca", 
                 plotter:str = "seaborn",
                 outdir:str = "output"
                ):
        """
        Initialize the Dimensionality_Reduction class with the provided data and parameters.

        Parameters:
        data (pd.DataFrame): The dataset containing features and labels.
        labels_column (str): The column name in the dataset that contains the labels.
        dim (str): The dimensionality of the plot ('2D' or '3D').
        method (str): The dimensionality reduction method to use ('pca', 'tsne', or 'umap').
        plotter (str): The plotting library to use ('seaborn' or 'plotly').
        outdir (str): The directory where output plots will be saved.
        """

        self.data = data
        self.X = data.drop(labels_column, axis = 1)
        self.labels = data[labels_column]
        self.labels_column = labels_column
        self.dim = dim
        self.method = method
        self.plotter = plotter
        self.outdir = outdir
        self.analyses = {
            "2D" : {
                "tsne" : TSNE(n_components=2, random_state=42),
                "pca" : PCA(n_components=2, random_state=42),
                "umap" : UMAP(n_components=2, init='random', random_state=42)
            },
            "3D" : {
                "tsne" : TSNE(n_components=3, random_state=42),
                "pca" : PCA(n_components=3, random_state=42),
                "umap" : UMAP(n_components=3, init='random', random_state=42)
            }
        }
        os.makedirs(outdir, exist_ok=True)

    def plot2Dscatter(self, method:str = None, plotter:str = None):
        """
        Generate a 2D scatter plot using PCA, t-SNE, or UMAP.

        Parameters:
        method (str): The dimensionality reduction method to use ('pca', 'tsne', or 'umap').
        plotter (str): The plotting library to use ('seaborn' or 'plotly').

        This method saves the generated plot as both PNG and PDF files in the output directory.
        """

        try:
            if type(method) == type(None):
                method = self.method
            if type(plotter) == type(None):
                plotter = self.plotter
    
            # perform analysis
            logging.info(f"Performing 2D {method.upper()} Analysis")
            components = self.analyses["2D"][method].fit_transform(self.X)
    
            if method == "umap":
                components = pd.DataFrame(components, columns = ["umap0","umap1"])
    
            # plot with seaborn
            logging.info(f"Plotting 2D {method.upper()} Plots with {plotter}")
            if plotter == "seaborn":
                sns.scatterplot(data=pd.concat([components, self.labels]), 
                                x=f"{method}0", y=f"{method}1", hue=self.labels_column, style=self.labels_column, s = 128);
                
                plt.xlabel(f"{method}0",fontsize=20)
                plt.ylabel(f"{method}1",fontsize=20)
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)
                plt.legend(fontsize=20)
                plt.tight_layout()
    
                plt.savefig(f'{self.outdir}/2D_{method}_plot.png')
                plt.savefig(f'{self.outdir}/2D_{method}_plot.pdf')
                plt.show()
    
            # plot with plotly
            elif plotter == "plotly":
                fig = px.scatter( components, x=f"{method}0", y=f"{method}1", 
                                 color=self.labels, labels={'color': self.labels_column})
                fig.update_layout(height=500, width = 800)
                fig.update_traces(marker=dict(size=10))
                fig.write_image(f'{self.outdir}/2D_{method}_plot.png')
                fig.write_image(f'{self.outdir}/2D_{method}_plot.pdf')
                fig.show()

        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e, sys)
    
    def plot3Dscatter(self, method:str = None, plotter:str = None):
        """
        Generate a 3D scatter plot using PCA, t-SNE, or UMAP.

        Parameters:
        method (str): The dimensionality reduction method to use ('pca', 'tsne', or 'umap').
        plotter (str): The plotting library to use ('seaborn' or 'plotly').

        This method saves the generated plot as both PNG and PDF files in the output directory.
        """

        try:
            if type(method) == type(None):
                method = self.method
            if type(plotter) == type(None):
                plotter = self.plotter
    
            # perform analysis
            logging.info(f"Performing 3D {method.upper()} Analysis")
            components = self.analyses["3D"][method].fit_transform(self.X)
    
            if method == "umap":
                components = pd.DataFrame(components, columns = ["umap0","umap1", "umap2"])
    
            logging.info(f"Plotting 3D {method.upper()} Plots with {plotter}")
            # plot with seaborn
            if plotter == "seaborn":
                # Create a 3D scatter plot
                fig = plt.figure(figsize=(15,15))
                ax = plt.axes(projection ="3d")
                
                # Define color palette
                palette = sns.color_palette("husl", len(set(self.labels)))
                color_map = dict(zip(set(self.labels), palette))
                
                # Plotting
                for label in set(self.labels):
                    mask = [lbl == label for lbl in self.labels]
                    ax.scatter(components.loc[mask, f'{method}0'], 
                               components.loc[mask, f'{method}1'], 
                               components.loc[mask, f'{method}2'], 
                               label=label, 
                               color=color_map[label], 
                               s=128) 
                
                # Labels and title
                ax.set_xlabel(f'{method}0')
                ax.set_ylabel(f'{method}1')
                ax.set_zlabel(f'{method}2')
                ax.set_title(f'3D {method.upper()} Plot')
                
                # Legend
                ax.legend(title=self.labels_column)
                
                # Save plot
                plt.savefig(f'{self.outdir}/3D_{method}_plot.png')
                plt.savefig(f'{self.outdir}/3D_{method}_plot.pdf')
                plt.show()
    
    
            # plot with plotly
            elif plotter == "plotly":
                fig = px.scatter_3d(
                    components, x=f"{method}0", y=f"{method}1", z=f"{method}2",
                    color=self.labels, labels={'color': self.labels_column}
                )
                fig.update_traces(marker_size=8)
                fig.update_layout(height=800, width = 900)
                # html file
                plotly.offline.plot(fig, filename=f'{self.outdir}/3D_{method}_plot.html')
                fig.show()

        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e, sys)

    def runPlots(self, runs:list = ["2d_pca", "2d_tsne", "2d_umap"]):
        """
        Execute a series of dimensionality reduction analyses and generate the corresponding plots.

        Parameters:
        runs (list): A list of strings specifying the analyses to run. Each string should be in the 
                     format 'plot_type_method' (e.g., '2d_pca', '3d_tsne').
        
        The method iterates over the specified analyses, performs the dimensionality reduction,
        and generates the respective plots.
        """

        try:
            length = 110
            print("="*length)
            print(" Starting Dimensionality Reduction Analysis ")
            print("="*length)
            logging.info(f"RUNNING DIMENSIONALITY REDUCTION ANALYSES: {runs}")
            # iterate over runs
            for run in runs:
                # get plot type and method
                plot_type = run.split("_")[0]
                method = run.split("_")[1]
    
                print("="*length)
                print(f" Starting {plot_type.upper()} {method.upper()} Analysis ")
                print("="*length)
    
                if plot_type == "2d":
                    self.plot2Dscatter(method = method)
                else:
                    self.plot2Dscatter(method = method)
    
            print("="*length)
            print(" Dimensionality Reduction Analysis Completed ")
            print("="*length)
            logging.info(f"COMPLETED DIMENSIONALITY REDUCTION ANALYSES")

        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e, sys)
