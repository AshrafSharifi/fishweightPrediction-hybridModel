import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Custom_plots:
    def __init__(self, predicted_values, actual_values, writer=None, title="", summarytitle=""):
        self.actual_values = actual_values
        self.predicted_values = predicted_values
        self.writer = writer
        self.title = title
        self.summarytitle = summarytitle


    def plot_predictions(self, ax):
        ax.plot(self.actual_values, label="Actual Values", color="blue", alpha=0.6, linewidth=2)
        ax.plot(self.predicted_values, label="Predicted Values", color="orange", linestyle="--", alpha=0.7)
        ax.set_xlabel("Index of Test Samples", fontsize=14)
        ax.set_ylabel("Actual Fish weights", fontsize=14)
        ax.set_title("Predicted vs Actual Values", fontsize=14, fontweight="bold")
        ax.legend()

    def plot_actual_vs_predicted_scatter(self, ax):
        ax.scatter(self.predicted_values, self.actual_values, alpha=0.6, edgecolors="k", label="Predictions")
        ax.plot(
            [min(self.actual_values), max(self.actual_values)], 
            [min(self.actual_values), max(self.actual_values)], 
            color="red", linestyle="--", linewidth=2, label="Perfect Fit"
        )
        ax.set_xlabel("Predicted Fish weights", fontsize=14)
        ax.set_ylabel("Actual Fish weights", fontsize=14)
        ax.set_title("Scatter plot: Actual vs. Predicted Values", fontsize=14, fontweight="bold")
        ax.legend()

    def plot_residuals(self, ax):
        residuals = self.actual_values - self.predicted_values
        ax.scatter(range(len(residuals)), residuals, color="purple", alpha=0.6, edgecolors="k")
        ax.axhline(0, color="red", linestyle="--", linewidth=2)
        ax.set_xlabel("Index of Test Samples", fontsize=14)
        ax.set_ylabel("Residuals (Actual - Predicted)", fontsize=14)
        ax.set_title("Residuals Plot", fontsize=14, fontweight="bold")

    def plot_predictions_with_hist(self, ax):
        bins = np.histogram_bin_edges(
            np.concatenate([self.actual_values, self.predicted_values]), bins=30
        )
        # Plot histograms
        ax.hist(self.actual_values, bins=bins, label="Actual Values", alpha=0.6, color="blue", edgecolor="black")
        ax.hist(self.predicted_values, bins=bins, label="Predicted Values", alpha=0.6, color="orange", edgecolor="black")
        ax.set_title("Distribution of Predicted vs Actual Values", fontsize=14, fontweight="bold")
        ax.set_xlabel("Actual Fish Weights", fontsize=14)
        ax.set_ylabel("Count", fontsize=14)
        ax.legend(loc="best", fontsize=10)  # Ensure the legend is placed appropriately
        ax.grid(alpha=0.3)  # Add a light grid for better readability

    def plot_all(self):
        # Create a 2x2 grid of subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        plt.tight_layout(pad=5.0)

        # Generate each plot on the respective subplot
        self.plot_predictions(axs[0, 0])
        self.plot_actual_vs_predicted_scatter(axs[0, 1])
        self.plot_residuals(axs[1, 0])
        self.plot_predictions_with_hist(axs[1, 1])

        # Add a global title for the entire figure
        fig.suptitle(self.title if self.title else "Performance Analysis of Predictions", 
                     fontsize=16, fontweight="bold", color="red")

   

        # Adjust layout to fit the title and labels
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Display the entire figure
        plt.show()

        # Log the figure to TensorBoard if writer is provided
        if self.writer is not None:
            self.writer.add_figure("All Plots_" + self.summarytitle, fig)


# Example usage:
# predicted_values = np.random.rand(100)
# actual_values = np.random.rand(100)
# writer = SummaryWriter()  # Uncomment if you have a TensorBoard writer
# plots = Custom_plots(predicted_values, actual_values, writer)
# plots.plot_all()
# writer.close()
