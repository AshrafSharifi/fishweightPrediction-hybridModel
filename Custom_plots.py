import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Custom_plots:
    def __init__(self,predicted_values,actual_values,writer=None):
        self.actual_values= actual_values
        self.predicted_values= predicted_values
        self.Writer = writer
    
    
    def plot_predictions(self):
        fig = plt.figure(figsize=(12, 6))
        plt.plot(self.actual_values, label="Actual Values", color="blue", alpha=0.6, linewidth=2)
        plt.plot(self.predicted_values, label="Predicted Values", color="orange", linestyle="--", alpha=0.7)
        plt.xlabel("Samples")
        plt.ylabel("Target Variable")
        plt.title("Predicted vs Actual Values")
        plt.legend()
        plt.show()
        if self.Writer!=None:
            self.Writer.add_figure("Prediction vs Actual", fig)
        
    def plot_actual_vs_predicted_scatter(self):
        fig = plt.figure(figsize=(8, 8))
        plt.scatter(self.predicted_values, self.actual_values, alpha=0.6, edgecolors="k", label="Predictions")
        plt.plot([min(self.actual_values), max(self.actual_values)], [min(self.actual_values), max(self.actual_values)], 
                 color="red", linestyle="--", linewidth=2, label="Perfect Fit")
        plt.xlabel("Predicted Values")
        plt.ylabel("Actual Values")
        plt.title("Scatter plot: Actual vs. Predicted Values")
        plt.legend()
        plt.show()
        if self.Writer!=None:
            self.Writer.add_figure("Actual vs. Predicted Values", fig)
        
    def plot_residuals(self):
        residuals = self.actual_values - self.predicted_values
        fig = plt.figure(figsize=(10, 6))
        plt.scatter(range(len(residuals)), residuals, color="purple", alpha=0.6, edgecolors="k")
        plt.axhline(0, color="red", linestyle="--", linewidth=2)
        plt.xlabel("Samples")
        plt.ylabel("Residuals (Actual - Predicted)")
        plt.title("Residuals Plot")
        plt.show()
        if self.Writer!=None:
            self.Writer.add_figure("Residuals Plot", fig)
        
    def plot_predictions_with_hist(self):
        fig, ax = plt.subplots(figsize=(15, 6))   
        # Histogram for distribution comparison
        sns.histplot(self.actual_values, label="Actual Values", color="blue", kde=True,  alpha=0.6)
        sns.histplot(self.predicted_values, label="Predicted Values", color="orange", kde=True, alpha=0.6)
        plt.title("Distribution of Predicted vs Actual Values")
        plt.xlabel("Value")
        plt.ylabel("Count")
        plt.legend()
        plt.show()
        if self.Writer!=None:
            self.Writer.add_figure("Distribution of Predicted vs Actual Values", fig)