# -*- coding: utf-8 -*-

from google.colab import drive
drive.mount('/content/drive')

pip install ipywidgets

import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import datetime
import json
import  os
from scipy.spatial import ConvexHull
from matplotlib.path import Path
import matplotlib.patches as patches
from datetime import datetime
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Screen dimensions
screen_width = 1920
screen_height = 1080
left_rectangle_x = 500
left_rectangle_y = 90
rectangle_width = 285
rectangle_height = 250
right_rectangle_x = 1120
right_rectangle_y = 90

"""# Cluster on Raw Gaze Data to know whether eye gaze is in which AOI"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull, QhullError

def preprocess(df_LVL1):
    df_LVL1[['EyeTracker-x', 'EyeTracker-y']] = df_LVL1['EyeTracker'].str.extract(r'\(\s*([-+]?\d*\.\d+|\d+)\s*,\s*([-+]?\d*\.\d+|\d+)\s*\)')

    df_LVL1['EyeTracker-x'] = pd.to_numeric(df_LVL1['EyeTracker-x'], errors='coerce').fillna(0.0)
    df_LVL1['EyeTracker-y'] = pd.to_numeric(df_LVL1['EyeTracker-y'], errors='coerce').fillna(0.0)

    df_LVL1 = df_LVL1[(df_LVL1['EyeTracker-x'] != 0) & (df_LVL1['EyeTracker-y'] != 0) &
                      (df_LVL1['EyeTracker-x'] != -1) & (df_LVL1['EyeTracker-y'] != -1)]

    df_LVL1 = df_LVL1[['Timestamp', 'EyeTracker-x', 'EyeTracker-y']]

    return df_LVL1

# Function to cluster EyeTracker_x and EyeTracker_y
def cluster_coordinates(df):
    """Cluster the EyeTracker_x and EyeTracker_y columns."""
    X = df[['EyeTracker-x', 'EyeTracker-y']]
    kmeans = KMeans(n_clusters=4, n_init='auto', random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)
    return df

# Function to count the instances in each cluster
def count_clusters(df):
    """Count the number of instances in each cluster."""
    cluster_counts = df['Cluster'].value_counts()
    return cluster_counts

# Updated plot_clusters function to ensure consistent color across plots
def plot_clusters(ax, df_clustered, screen_width, screen_height):
    """Plot clusters with convex hulls, ensuring consistent colors across plots."""
    # Define fixed colors for each cluster index (up to 7 clusters)
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # List of colors for clusters C1, C2, etc.

    for cluster in df_clustered['Cluster'].unique():
        cluster_points = df_clustered[df_clustered['Cluster'] == cluster][['EyeTracker-x', 'EyeTracker-y']].values
        color = colors[cluster % len(colors)]  # Assign fixed color from the list
        cluster_label = f'C{cluster + 1}'  # Naming clusters as C1, C2, C3...

        if len(cluster_points) < 3:  # Cannot compute hull with fewer than 3 points
            ax.plot(cluster_points[:, 0], cluster_points[:, 1], marker='.', linestyle='None', color=color, label=f'{cluster_label}')
            continue

        try:
            # Try to compute the convex hull
            hull = ConvexHull(cluster_points)
            ax.plot(cluster_points[:, 0], cluster_points[:, 1], marker='.', linestyle='None', color=color, label=f'{cluster_label}')

            # Plot the hull edges
            for simplex in hull.simplices:
                ax.plot(cluster_points[simplex, 0], cluster_points[simplex, 1], color=color, alpha=0.5)

            # Calculate and mark the centroid of the cluster
            centroid_x = cluster_points[:, 0].mean()
            centroid_y = cluster_points[:, 1].mean()
            ax.text(centroid_x, centroid_y, cluster_label, fontsize=15, color='black', fontweight='bold', ha='center', va='center')

        except QhullError:
            # Handle cases where points cannot form a valid convex hull
            print(f"Cluster {cluster}: Unable to compute convex hull, points may be collinear or insufficient.")
            ax.plot(cluster_points[:, 0], cluster_points[:, 1], marker='.', linestyle='None', color=color, label=f'{cluster_label} (no hull)')

def process_gaze_data_in_folder(folder_path, bg_image_path):
    # List all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # Ensure there are CSV files in the folder
    if not csv_files:
        print("No CSV files found in the folder.")
        return

    # Screen dimensions for plotting
    screen_width = 1920
    screen_height = 1080
    bg_image = mpimg.imread(bg_image_path)

    # Process each CSV file
    for file_name in csv_files:
        file_path = os.path.join(folder_path, file_name)

        # Read the CSV file
        df = pd.read_csv(file_path, sep=';')
        df_preprocessed = preprocess(df)

        if df_preprocessed.empty:
            print(f"No valid gaze data for file {file_name} after preprocessing. Skipping clustering and plotting.")
            continue  # Skip to the next file if there's no valid data

        # Cluster the coordinates
        df_clustered = cluster_coordinates(df_preprocessed)

        # Count clusters
        cluster_counts_sorted = count_clusters(df_clustered).sort_values(ascending=False)
        output_clusters = "[" + ">".join([f'C{int(i)+1}' for i in cluster_counts_sorted.index.astype(str)]) + "]"

        # Print the most prominent cluster
        max_cluster = cluster_counts_sorted.idxmax()
        print(f"For file {file_name}, the most prominent cluster is: C{max_cluster + 1}")

        # Plot the clusters
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.imshow(bg_image, extent=[0, screen_width, 0, screen_height], alpha=0.9)
        plot_clusters(ax, df_clustered, screen_width, screen_height)

        ax.set_title(f'Clusters for {file_name}\n{output_clusters}', fontsize=14)
        ax.set_xlabel('Gaze X')
        ax.set_ylabel('Gaze Y')
        ax.set_xlim(0, screen_width)
        ax.set_ylim(0, screen_height)
        ax.grid(True)

        # Show the plot
        plt.tight_layout()
        plt.show()

# Folder containing the CSV files
folder_path = '/content/drive/MyDrive/ET Dataset/PT/NEUROTYPICAL_GAZE_DATA_ATTENTION'
bg_image_path = '/content/drive/MyDrive/ET Dataset/Picture1.png'
process_gaze_data_in_folder(folder_path, bg_image_path)

"""#Are there differences in fixation patterns between simple and complex game tasks for NDD children?

# on full data
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import matplotlib.patches as patches

def analyze_fixations(file_path_LVL1, bg_image_path):
    def preprocessdataset(merged_ndddata):
        merged_ndddata.drop(index=merged_ndddata.index[:2], inplace=True)
        merged_ndddata.reset_index(drop=True, inplace=True)
        string_to_delete0 = ','
        merged_ndddata = merged_ndddata[~merged_ndddata.astype(str).apply(lambda x: x.str.contains(string_to_delete0)).any(axis=1)]
        merged_ndddata.columns = merged_ndddata.columns.str.strip()

        return merged_ndddata

    df_LVL1 = pd.read_csv(file_path_LVL1)
    df_LVL1 = preprocessdataset(df_LVL1)

    bg_image = mpimg.imread(bg_image_path)

    def plot_density(ax, x, y, title):
        ax.imshow(bg_image, extent=[0, screen_width, 0, screen_height], alpha=0.9)
        ax.scatter(x, y, s=30, alpha=0.9, color='black')
        ax.set_title(title)
        ax.set_xlabel('Fixation X')
        ax.set_ylabel('Fixation Y')
        ax.set_xlim(0, screen_width)
        ax.set_ylim(0, screen_height)
        ax.grid(True)

        # Plot left and right rectangles
        left_rectangle = patches.Rectangle((left_rectangle_x, left_rectangle_y), rectangle_width, rectangle_height,
                                           linewidth=3, edgecolor='yellow', facecolor='none')
        right_rectangle = patches.Rectangle((right_rectangle_x, right_rectangle_y), rectangle_width, rectangle_height,
                                            linewidth=3, edgecolor='yellow', facecolor='none')
        ax.add_patch(left_rectangle)
        ax.add_patch(right_rectangle)

    # Divide the screen into quadrants and count gaze points in each quadrant
    def count_quadrants(df):
        Q1_points = sum((df['fixation_x'] <= screen_width / 2) & (df['fixation_y'] >= screen_height / 2))
        Q2_points = sum((df['fixation_x'] > screen_width / 2) & (df['fixation_y'] >= screen_height / 2))
        Q3_points = sum((df['fixation_x'] <= screen_width / 2) & (df['fixation_y'] < screen_height / 2))
        Q4_points = sum((df['fixation_x'] > screen_width / 2) & (df['fixation_y'] < screen_height / 2))
        return {'Q1': Q1_points, 'Q2': Q2_points, 'Q3': Q3_points, 'Q4': Q4_points}

    # Group by studentID, activityID, and level
    for (student_id, activity_id, level), group_data in df_LVL1.groupby(['studentID', 'activityID', 'level']):
        group_data = group_data[(group_data['fixation_x'] > 0) & (group_data['fixation_y'] > 0)]

        # Create a new figure for each (studentID, activityID, level) combination
        fig, ax = plt.subplots(figsize=(15, 5))

        # Plot density for the current group (studentID, activityID, and level)
        title = f'Student: {student_id}, Activity: {activity_id}, Level: {level}'
        plot_density(ax, group_data['fixation_x'], group_data['fixation_y'], title)

        # Annotate quadrants on the plot
        ax.text(screen_width / 4, 3 * screen_height / 4, 'Q1', fontsize=12, color='blue')
        ax.text(3 * screen_width / 4, 3 * screen_height / 4, 'Q2', fontsize=12, color='blue')
        ax.text(screen_width / 4, screen_height / 4, 'Q3', fontsize=12, color='blue')
        ax.text(3 * screen_width / 4, screen_height / 4, 'Q4', fontsize=12, color='blue')

        # Display the plot
        plt.tight_layout()
        plt.show()

        # Save the plot to the output directory with studentID, activityID, and level in the filename
        #output_filepath = os.path.join(output_dir, f'{student_id}_activity_{activity_id}_level_{level}.png')
        #plt.savefig(output_filepath)
        #plt.close()

        # Count quadrants for the current (studentID, activityID, level)
        quadrants = count_quadrants(group_data)
        max_quadrant = max(quadrants, key=quadrants.get)

        # Output the quadrant with the most fixations for this (studentID, activityID, level)
        print(f"For student {student_id}, activity {activity_id}, level {level}, most fixations were in quadrant: {max_quadrant}")

# Provide file paths, background image path, and output directory
file_path_LVL1 = '/content/drive/MyDrive/ET Dataset/PT/ET_data_neurotypical (1).csv'
bg_image_path = '/content/drive/MyDrive/ET Dataset/Picture1.png'
#output_dir = '/content/drive/MyDrive/ET Dataset/Output_Plots/'

# Call the function
analyze_fixations(file_path_LVL1, bg_image_path)

"""# Line Graph for Fixations if they are on AOI"""

import pandas as pd
import matplotlib.pyplot as plt

def analyze_fixations(file_path_LVL1, student_id_to_filter, output_csv_path, cleaned_csv_path):
    # Preprocess the dataset to remove unnecessary rows and clean it
    def preprocessdataset(merged_ndddata):
        merged_ndddata.drop(index=merged_ndddata.index[:2], inplace=True)
        merged_ndddata.reset_index(drop=True, inplace=True)
        string_to_delete0 = ','
        merged_ndddata = merged_ndddata[~merged_ndddata.astype(str).apply(lambda x: x.str.contains(string_to_delete0)).any(axis=1)]
        merged_ndddata.columns = merged_ndddata.columns.str.strip()  # Clean column names
        return merged_ndddata

    # Load the dataset
    df_LVL1 = pd.read_csv(file_path_LVL1)
    df_LVL1 = preprocessdataset(df_LVL1)

    # Convert the 'timestamp' column to datetime
    df_LVL1['timestamp'] = pd.to_datetime(df_LVL1['timestamp'], errors='coerce')

    # Drop rows with any missing (NaN) values in the specific columns
    df_LVL1.dropna(subset=['fixation_x', 'fixation_y', 'timestamp'], inplace=True)

    # Filter the DataFrame for the specific student ID
    student_data = df_LVL1[df_LVL1['studentID'] == student_id_to_filter]
    student_data = student_data[['fixation_x', 'fixation_y', 'timestamp']]

    # Select the first 300 rows
    student_data = student_data.head(300)

    # Save the cleaned and processed df_LVL1 to a CSV file
    student_data.to_csv(cleaned_csv_path, index=False)
    print(f"Cleaned DataFrame (first 300 rows) saved to {cleaned_csv_path}")


    # Initialize columns for left and right rectangles
    student_data['left_rectangle'] = 0
    student_data['right_rectangle'] = 0

    # Update the rectangle columns based on the new logic
    for index, row in student_data.iterrows():
        if (left_rectangle_x <= row['fixation_x'] <= left_rectangle_x + rectangle_width) and \
           (left_rectangle_y <= row['fixation_y'] <= left_rectangle_y + rectangle_height):
            student_data.at[index, 'left_rectangle'] = 1  # Inside left rectangle
            student_data.at[index, 'right_rectangle'] = 0  # Not inside right rectangle
        elif (right_rectangle_x <= row['fixation_x'] <= right_rectangle_x + rectangle_width) and \
             (right_rectangle_y <= row['fixation_y'] <= right_rectangle_y + rectangle_height):
            student_data.at[index, 'right_rectangle'] = 1  # Inside right rectangle
            student_data.at[index, 'left_rectangle'] = 0  # Not inside left rectangle
        else:
            student_data.drop(index, inplace=True)  # Skip those rows not in either rectangle

    # Save the filtered student_data DataFrame to a CSV file
    student_data.to_csv(output_csv_path, index=False)
    print(f"Filtered student DataFrame (with fixations) saved to {output_csv_path}")

    # Plotting the left_rectangle and right_rectangle graphs
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6))

    # Set a main title for the entire figure
    fig.suptitle(f'Fixations for Student ID: {student_id_to_filter} (First 300 Rows)', fontsize=14)

    # Plot for the left rectangle
    ax1.plot(student_data['timestamp'], student_data['left_rectangle'],
             drawstyle='steps-post', color='blue', label='Left Rectangle')
    ax1.set_ylim(-0.05, 1.05)  # Set y-axis limits to reduce vertical space
    ax1.set_yticks([0, 1])      # Set y-ticks to show only 0 and 1
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('Left Rectangle')
    ax1.grid()

    # Plot for the right rectangle
    ax2.plot(student_data['timestamp'], student_data['right_rectangle'],
             drawstyle='steps-post', color='green', label='Right Rectangle')
    ax2.set_ylim(-0.05, 1.05)  # Set y-axis limits to reduce vertical space
    ax2.set_yticks([0, 1])      # Set y-ticks to show only 0 and 1
    ax2.set_xlabel('Timestamp')
    ax2.set_ylabel('Right Rectangle')
    ax2.grid()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make space for the main title
    plt.show()

# Provide the file paths, student ID, and output CSV paths
file_path_LVL1 = '/content/drive/MyDrive/ET Dataset/PT/ET_data_neurotypical (1).csv'
student_id_to_filter = 'VA3_3'  # Student ID to filter
output_csv_path = '/content/drive/MyDrive/ET Dataset/PT/filtered_student_data_300rows.csv'  # Path to save filtered student data
cleaned_csv_path = '/content/drive/MyDrive/ET Dataset/PT/cleaned_LVL1_data_300rows.csv'  # Path to save cleaned df_LVL1

# Call the function to analyze fixations for the first 300 rows and save to CSV
analyze_fixations(file_path_LVL1, student_id_to_filter, output_csv_path, cleaned_csv_path)

"""#Heatmap codes"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import numpy as np

# Function to preprocess the dataset
def preprocessdataset(df):
    df.drop(index=df.index[:2], inplace=True)  # Drop the first two rows
    df.reset_index(drop=True, inplace=True)  # Reset index
    df.columns = df.columns.str.strip()  # Strip any extra spaces from column names
    df = df[df['fixation_x'] > 0]  # Keep only positive fixation_x values
    df = df[df['fixation_y'] > 0]  # Keep only positive fixation_y values
    return df

# Function to plot different density variants side by side for a specific studentID
def plot_fixations_for_student(df, bg_image, student_id, screen_width=1920, screen_height=1080):
    # Filter the dataframe for the given studentID
    student_data = df[df['studentID'] == student_id]

    # Group by 'activityID' and 'level' to plot different plots for the student
    for (activity_id, level), group_data in student_data.groupby(['activityID', 'level']):

        # Create a single row of subplots with 3 columns (excluding Attention Map)
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))  # 1 row, 2 columns for each plot

        # KDE Plot (Kernel Density Estimate) for Heatmap
        ax = axes[0]
        ax.imshow(bg_image, extent=[0, screen_width, 0, screen_height], alpha=0.9)
        sns.kdeplot(x=group_data['fixation_x'], y=group_data['fixation_y'], cmap="Reds", fill=True, bw_adjust=0.5, ax=ax)
        ax.set_title(f'KDE Heatmap')
        ax.set_xlim(0, screen_width)
        ax.set_ylim(0, screen_height)
        ax.set_xlabel('Fixation X')
        ax.set_ylabel('Fixation Y')

        # Contour Plot
        ax = axes[1]
        ax.imshow(bg_image, extent=[0, screen_width, 0, screen_height], alpha=0.9)
        sns.kdeplot(x=group_data['fixation_x'], y=group_data['fixation_y'], cmap="Greens", ax=ax, levels=10)
        ax.set_title(f'Contour Plot')
        ax.set_xlim(0, screen_width)
        ax.set_ylim(0, screen_height)
        ax.set_xlabel('Fixation X')
        ax.set_ylabel('Fixation Y')

        # Add a global title
        fig.suptitle(f'Student: {student_id}, Activity: {activity_id}, Level: {level}', fontsize=16)

        # Adjust layout for better fit
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the suptitle

        # Show the combined plot
        plt.show()

# Main function to process and plot fixations for the first student
def analyze_fixations(file_path_LVL1, bg_image_path):
    # Load and preprocess the dataset
    df_LVL1 = pd.read_csv(file_path_LVL1)
    df_LVL1 = preprocessdataset(df_LVL1)

    # Select the first student from the dataset
    first_student_id = df_LVL1['studentID'].unique()[1]  # Select the first unique student ID

    # Load background image
    bg_image = mpimg.imread(bg_image_path)

    # Plot different density plots for the first studentID
    plot_fixations_for_student(df_LVL1, bg_image, first_student_id)

file_path_LVL1 = '/content/drive/MyDrive/ET Dataset/PT/ET_data_neurotypical (1).csv'
bg_image_path = '/content/drive/MyDrive/ET Dataset/Picture1.png'

# Call the function to run for the first student
analyze_fixations(file_path_LVL1, bg_image_path)

"""Makign table for fixtion in q3 and q4"""

import pandas as pd

def preprocess_and_extract(file_path_LVL1):
    # Load the dataset
    df_LVL1 = pd.read_csv(file_path_LVL1)

    # Remove rows with NaN or 0 in studentID, activityID, activitytype, or level
    df_LVL1 = df_LVL1[~df_LVL1[['studentID', 'activityID', 'activitytype', 'level']].isnull().any(axis=1)]
    df_LVL1 = df_LVL1[(df_LVL1[['studentID', 'activityID', 'activitytype', 'level']] != 0).all(axis=1)]

    # Screen dimensions
    screen_width = 1920
    screen_height = 1080

    # Determine if fixations are in Q3 or Q4
    df_LVL1['is_in_q3_q4'] = (
        ((df_LVL1['fixation_x'] > screen_width / 2) & (df_LVL1['fixation_y'] < screen_height / 2)) |  # Q4
        ((df_LVL1['fixation_x'] <= screen_width / 2) & (df_LVL1['fixation_y'] >= screen_height / 2))   # Q3
    )

    return df_LVL1

# Provide file path
file_path_LVL1 = '/content/drive/MyDrive/ET Dataset/PT/ET_data_neurotypical (1).csv'

# Extract and preprocess the dataset
newdf = preprocess_and_extract(file_path_LVL1)

# Save the resulting dataframe to a CSV file
newdf.to_csv('/content/drive/MyDrive/ET Dataset/PT/is_in_q3_q4.csv', index=False)

# Display the first 100 rows of the resulting dataframe
newdf.head(100)

"""# Can fixation duration and location be used as indicators of cognitive load during gameplay for NDD children?"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function to preprocess the dataset
def preprocessdataset(df):
    # Drop rows with missing values in key columns
    essential_columns = ['studentID', 'activityID', 'level', 'fixation_x', 'fixation_y', 'fixation_duration']
    df = df.dropna(subset=essential_columns)

    # Convert relevant columns to appropriate data types using .loc
    df.loc[:, 'fixation_x'] = pd.to_numeric(df['fixation_x'], errors='coerce')
    df.loc[:, 'fixation_y'] = pd.to_numeric(df['fixation_y'], errors='coerce')
    df.loc[:, 'fixation_duration'] = pd.to_numeric(df['fixation_duration'], errors='coerce')

    # Drop any rows that could not be converted to numeric (if any)
    df = df.dropna(subset=['fixation_x', 'fixation_y', 'fixation_duration'])

    # Remove duplicates
    df = df.drop_duplicates()
    return df

# Function to plot fixation data
def plot_fixations(df, ax, title, bg_image, screen_width, screen_height, fig):
    ax.imshow(bg_image, extent=[0, screen_width, 0, screen_height], alpha=0.9)

    # Use fixation_duration to color the points
    sc = ax.scatter(df['fixation_x'], df['fixation_y'], s=df['fixation_duration'], alpha=0.9, c=df['fixation_duration'], cmap='plasma')

    ax.set_title(title)
    ax.set_xlabel('Fixation X')
    ax.set_ylabel('Fixation Y')
    ax.set_xlim(0, screen_width)
    ax.set_ylim(0, screen_height)
    ax.grid(True)

    # Adding the color bar now that we have varying colors
    fig.colorbar(sc, ax=ax, label='Fixation Duration (ms)')

    # Resetting the index for proper access
    df.reset_index(drop=True, inplace=True)

    # Finding and returning the highest fixation duration and its location
    max_duration_idx = np.argmax(df['fixation_duration'])
    max_duration = df.loc[max_duration_idx, 'fixation_duration']
    max_location = (df.loc[max_duration_idx, 'fixation_x'], df.loc[max_duration_idx, 'fixation_y'])
    return max_duration, max_location

# Main function
def plot_fixation_duration_by_student_activity(bg_image_path, csv_path):
    # Screen dimensions
    screen_width = 1920
    screen_height = 1080
    bg_image = plt.imread(bg_image_path)
    df = pd.read_csv(csv_path)
    df = preprocessdataset(df)

    # Grouping the DataFrame by 'studentID', 'activityID', and 'level'
    grouped = df.groupby(['studentID', 'activityID', 'level'])

    for (student_id, activity_id, level), group_df in grouped:
        if group_df.empty:
            print(f"No valid data for Student ID: {student_id}, Activity ID: {activity_id}, Level: {level}")
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        title = f'Student ID: {student_id}, Activity ID: {activity_id}, Level: {level}'

        max_duration, max_location = plot_fixations(group_df, ax, title, bg_image, screen_width, screen_height, fig)

        plt.tight_layout()
        plt.show()

        # Print the highest duration for the current group
        print(f"Highest fixation duration for Student ID: {student_id}, Activity ID: {activity_id}, Level: {level}: {max_duration} ms, at location {max_location}")

# Background image path for plotting
bg_image_path = '/content/drive/MyDrive/ET Dataset/Picture1.png'
# File path for the CSV (update as necessary)
file_path = '/content/drive/MyDrive/ET Dataset/PT/ET_data_neurotypical (1).csv'

# Call the function to analyze fixation duration for unique combinations of studentID, activityID, and level
plot_fixation_duration_by_student_activity(bg_image_path, file_path)
print("-" * 180)

"""Top 5 Durations for each children"""

import pandas as pd

# Function to preprocess the dataset
def preprocessdataset(df):
    # Drop rows with missing values in key columns
    essential_columns = ['studentID', 'activityID', 'level', 'fixation_x', 'fixation_y', 'fixation_duration']
    df = df.dropna(subset=essential_columns)

    # Convert relevant columns to appropriate data types
    df.loc[:, 'fixation_x'] = pd.to_numeric(df['fixation_x'], errors='coerce')
    df.loc[:, 'fixation_y'] = pd.to_numeric(df['fixation_y'], errors='coerce')
    df.loc[:, 'fixation_duration'] = pd.to_numeric(df['fixation_duration'], errors='coerce')

    # Drop any rows that could not be converted to numeric (if any)
    df = df.dropna(subset=['fixation_x', 'fixation_y', 'fixation_duration'])

    # Remove duplicates
    df = df.drop_duplicates()
    return df

# Function to extract the top 5 fixation durations for each studentID
def extract_top_fixations_by_student(file_path):
    # Load and preprocess the dataset
    df = pd.read_csv(file_path)
    df = preprocessdataset(df)

    # Group by 'studentID' and extract top 5 rows with highest fixation durations for each group
    top_fixations = df.groupby('studentID').apply(lambda x: x.nlargest(5, 'fixation_duration')).reset_index(drop=True)

    # Display the top 5 fixation durations for each student in a clear table format
    return top_fixations[['studentID', 'activityID', 'level', 'fixation_x', 'fixation_y', 'fixation_duration']]

# File path for the CSV
file_path = '/content/drive/MyDrive/ET Dataset/PT/ET_data_neurotypical (1).csv'

# Extract and display the top fixations
top_fixations_table = extract_top_fixations_by_student(file_path)
top_fixations_table.head(100)

"""# Extracting Reaction time, Sustained Attention"""

file_path_LVL1 = '/content/drive/MyDrive/ET Dataset/PT/Student3_Session4_Activity2_ATTENTION.csv'
df_LVL1 = pd.read_csv(file_path_LVL1,sep=';')
df_LVL1.head()

import pandas as pd
import re

def preprocess(df_LVL1):
    # Split EyeTracker into EyeTracker-x and EyeTracker-y using a more flexible regex
    df_LVL1[['EyeTracker-x', 'EyeTracker-y']] = df_LVL1['EyeTracker'].str.extract(r'\(\s*([-+]?\d*\.\d+|\d+)\s*,\s*([-+]?\d*\.\d+|\d+)\s*\)')

    # Convert extracted values to float type, fill NaNs with 0.0
    df_LVL1['EyeTracker-x'] = pd.to_numeric(df_LVL1['EyeTracker-x'], errors='coerce').fillna(0.0)
    df_LVL1['EyeTracker-y'] = pd.to_numeric(df_LVL1['EyeTracker-y'], errors='coerce').fillna(0.0)

    # Drop rows where EyeTracker-x or EyeTracker-y are negative
    df_LVL1 = df_LVL1[(df_LVL1['EyeTracker-x'] >= 0) & (df_LVL1['EyeTracker-y'] >= 0)]

    # Drop the old EyeTracker column
    df_LVL1.drop(columns=['EyeTracker'], inplace=True)

    # Reorder the columns to place EyeTracker_x and EyeTracker_y at the 2nd and 3rd positions
    columns_order = ['Timestamp', 'EyeTracker-x', 'EyeTracker-y'] + [col for col in df_LVL1.columns if col not in ['Timestamp', 'EyeTracker-x', 'EyeTracker-y']]
    df_LVL1 = df_LVL1[columns_order]

    # Create new columns for obj_x, obj_y, obj_z with NaN values as default
    df_LVL1['obj_x'] = pd.NA
    df_LVL1['obj_y'] = pd.NA
    df_LVL1['obj_z'] = pd.NA

    # Function to process the Message column
    def extract_coordinates(aux_message):
        # Ensure aux_message is a string, if not return NaN for coordinates
        if isinstance(aux_message, str):
            # Check if '_(' is in the aux_message and extract accordingly
            if '_(' in aux_message:
                # Extract the part before '_(' and coordinates after '_('
                message_part = aux_message.split('_(')[0]
                coord_part = aux_message.split('_(')[1].replace(')', '')  # remove the closing parenthesis

                # Split the coordinate part into three values
                coords = re.split(r'\s+', coord_part.strip())
                if len(coords) == 3:
                    obj_x, obj_y, obj_z = coords
                else:
                    obj_x = obj_y = obj_z = pd.NA
                return message_part, obj_x, obj_y, obj_z
        # Return original message and NaNs if not a valid string
        return aux_message, pd.NA, pd.NA, pd.NA

    # Apply the function to each row in the Message column
    df_LVL1[['Message', 'obj_x', 'obj_y', 'obj_z']] = df_LVL1['Message'].apply(
        lambda x: pd.Series(extract_coordinates(x))
    )

    # Remove any commas from string-type columns
    df_LVL1 = df_LVL1.applymap(lambda x: x.replace(',', '') if isinstance(x, str) else x)

    # Calculate the percentage of 0 or NaN values in EyeTracker_x and EyeTracker_y
    for col in ['EyeTracker-x', 'EyeTracker-y']:
        zero_nan_count = df_LVL1[col].isna().sum() + (df_LVL1[col] == 0).sum()
        total_count = df_LVL1[col].shape[0]
        percentage = (zero_nan_count / total_count) * 100

        # Check if the percentage is greater than 50%
        if percentage > 50:
            print(f"The feature '{col}' has {percentage:.2f}% of its values as 0 or NaN.")

    return df_LVL1

newdata= preprocess(df_LVL1)
newdata.head()

newdata.to_csv('/content/drive/MyDrive/ET Dataset/PT/Newdata.csv',index=False)

import pandas as pd

# File path to the CSV data
data_path = '/content/drive/MyDrive/ET Dataset/PT/Newdata.csv'

# Read the CSV file
df = pd.read_csv(data_path)

# Check if the 'Message' column exists, and print unique values
if 'Message' in df.columns:
    unique_messages = df['Message'].unique()
    print("Unique values in the 'Message' column:")
    for message in unique_messages:
        print(message)
else:
    print("'Message' column not found in the dataset.")

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime

# Extracting Reaction time, Sustained Attention
def sustained_attention(data_path, bg_image_path, output_path):
    # Initialize variables
    yellow_purple_active = False
    n_hit = 0
    n_et_hit = 0
    e_start = None
    e_end = None
    Appear_Distractor_FirstButterfly_list = []

    # Initialize old value variable
    old_value = None
    new_value = None
    # Initialize counters for fixations
    total_fixations = 0
    fixations_in_rectangle = 0
    fixations_outside_rectangle = 0

    # Load DataFrame
    df = pd.read_csv(data_path)
    df_copy = df.copy()

    # Add new columns
    df_copy['quadrant'] = ""
    df_copy['AOI status'] = ""
    df_copy['object'] = ""
    df_copy['Engagement'] = ""

    # Screen dimensions
    screen_width = 1960
    screen_height = 1080

    # Rectangle coordinates
    left_rectangle_x = 480
    left_rectangle_y = 105
    rectangle_width = 285
    rectangle_height = 250

    right_rectangle_x = screen_width - rectangle_width - 500
    right_rectangle_y = 110

    # Filter data where EyeTracker-x and EyeTracker-y are positive
    df_filtered = df[(df['EyeTracker-x'] > 0) & (df['EyeTracker-y'] > 0)]

    plt.figure(figsize=(8, 6))
    plt.xlim(0, screen_width)
    plt.ylim(0, screen_height)

    bg_image = plt.imread(bg_image_path)
    plt.imshow(bg_image, extent=[0, screen_width, 0, screen_height])

    # Plot left rectangle
    left_rectangle = patches.Rectangle((left_rectangle_x, left_rectangle_y), rectangle_width, rectangle_height, linewidth=3, edgecolor='yellow', facecolor='none')
    plt.gca().add_patch(left_rectangle)

    # Plot right rectangle
    right_rectangle = patches.Rectangle((right_rectangle_x, right_rectangle_y), rectangle_width, rectangle_height, linewidth=3, edgecolor='yellow', facecolor='none')
    plt.gca().add_patch(right_rectangle)

    # Plot the first point
    plt.scatter(646.0, 234.0, color='pink', label='Point 1')

    # Plot the second point
    plt.scatter(1326.0, 234.0, color='blue', label='Point 2')

    plt.scatter(df_filtered['EyeTracker-x'], df_filtered['EyeTracker-y'], color='black', s=10)

    # Annotate quadrants on the plot
    plt.text(screen_width / 4, 3 * screen_height / 4, 'Q1', fontsize=12, color='blue')
    plt.text(3 * screen_width / 4, 3 * screen_height / 4, 'Q2', fontsize=12, color='blue')
    plt.text(screen_width / 4, screen_height / 4, 'Q3', fontsize=12, color='blue')
    plt.text(3 * screen_width / 4, screen_height / 4, 'Q4', fontsize=12, color='blue')

    plt.xlabel('EyeTracker-x')
    plt.ylabel('EyeTracker-y')
    plt.title('Eye Tracking Data with Quadrants')
    plt.grid(True)
    plt.show()

    # List of specific old message values to check against
    specific_old_messages = [
        "Disappear_Distractor_FirstButterfly",
        "Disappear_Distractor_Branch",
        "Disappear_Distractor_SecondButterfly",
        "Disappear_Distractor_Trunk",
        "Disappear_Distractor_BlueFlowers",
        "Disappear_Target_Mushroom",
        "Neutral_Click"
    ]

    for index, row in df_filtered.iterrows():
        total_fixations += 1  # Increment total fixations
        # Determine the quadrant and whether the fixations are in the rectangles
        if row['EyeTracker-x'] <= screen_width / 2 and row['EyeTracker-y'] >= screen_height / 2:
            quadrant = "Q1"
            in_rectangle = False
        elif row['EyeTracker-x'] > screen_width / 2 and row['EyeTracker-y'] >= screen_height / 2:
            quadrant = "Q2"
            in_rectangle = False
        elif row['EyeTracker-x'] <= screen_width / 2 and row['EyeTracker-y'] < screen_height / 2:
            quadrant = "Q3"
            if left_rectangle_x <= row['EyeTracker-x'] <= left_rectangle_x + rectangle_width and left_rectangle_y <= row['EyeTracker-y'] <= left_rectangle_y + rectangle_height:
                in_rectangle = True
                fixations_in_rectangle += 1  # Increment fixations inside rectangle
            else:
                in_rectangle = False
        elif row['EyeTracker-x'] > screen_width / 2 and row['EyeTracker-y'] < screen_height / 2:
            quadrant = "Q4"
            if right_rectangle_x <= row['EyeTracker-x'] <= right_rectangle_x + rectangle_width and right_rectangle_y <= row['EyeTracker-y'] <= right_rectangle_y + rectangle_height:
                in_rectangle = True
                fixations_in_rectangle += 1  # Increment fixations inside rectangle
            else:
                in_rectangle = False
        if not in_rectangle:
            fixations_outside_rectangle += 1  # Increment fixations outside rectangle

        # Check and update the old_value based on the current message
        currentmessage = row['Message']
        if pd.isnull(currentmessage) or currentmessage == "nan":
            if old_value is None:
                old_value = "No Stimuli"  # Initialize if it's the first row
        else:
            # Handle message updates
            if currentmessage == "Appear_Distractor_FirstButterfly":
                old_value = "Appear_Distractor_FirstButterfly"
                n_et_hit += 1
            elif currentmessage == "Disappear_Distractor_FirstButterfly":
                old_value = "Disappear_Distractor_FirstButterfly"  # Reset after disappearance
            elif currentmessage == "Appear_Distractor_Branch":
                old_value = "Appear_Distractor_Branch"
            elif currentmessage == "Disappear_Distractor_Branch":
                old_value = "Disappear_Distractor_Branch"  # Reset after disappearance
            elif currentmessage == "Appear_Distractor_SecondButterfly":
                old_value = "Appear_Distractor_SecondButterfly"
            elif currentmessage == "Disappear_Distractor_SecondButterfly":
                old_value = "Disappear_Distractor_SecondButterfly"  # Reset after disappearance
            elif currentmessage == "Appear_Distractor_Trunk":
                old_value = "Appear_Distractor_Trunk"
            elif currentmessage == "Disappear_Distractor_Trunk":
                old_value = "Disappear_Distractor_Trunk"  # Reset after disappearance
            elif currentmessage == "Appear_Distractor_BlueFlowers":
                old_value = "Appear_Distractor_BlueFlowers"
            elif currentmessage == "Disappear_Distractor_BlueFlowers":
                old_value = "Disappear_Distractor_BlueFlowers"  # Reset after disappearance
            elif currentmessage == "Appear_Target_Mushroom":
                old_value = "Appear_Target_Mushroom"
            elif currentmessage == "Disappear_Target_Mushroom":
                old_value = "Disappear_Target_Mushroom"  # Reset after disappearance

        # Update new columns with extracted values
        df_copy.loc[index, 'quadrant'] = quadrant
        df_copy.loc[index, 'AOI status'] = 'In AoI' if in_rectangle else 'Not in AoI'

        # Check if current message is null or "nan", and if old_message is in the specified list
        if (pd.isnull(currentmessage) or currentmessage == "nan") and old_value in specific_old_messages:
            df_copy.loc[index, 'object'] = currentmessage
        else:
            df_copy.loc[index, 'object'] = old_value

        # Update the Engagement column based on conditions
        if df_copy.loc[index, 'AOI status'] == 'In AoI' and 'appear' in str(old_value).lower():
            df_copy.loc[index, 'Engagement'] = 'Yes'
        else:
            df_copy.loc[index, 'Engagement'] = 'No'  # Set to 'No' if conditions are not met

    # Track each unique 'appear' event where Engagement is 'Yes'
    object_counts = {}

    for index, row in df_copy.iterrows():
        # Check if 'object' is a string and starts with "Appear", and if 'Engagement' is "Yes"
        if isinstance(row['object'], str) and row['object'].startswith("Appear") and row['Engagement'] == "Yes":
            object_type = row['object']
            if object_type not in object_counts:
                object_counts[object_type] = 0
            object_counts[object_type] += 1


    # Rearrange columns after adding new data
    column_order = [
        'Timestamp',
        'EyeTracker-x',
        'EyeTracker-y',
        'Message',
        'quadrant',
        'AOI status',
        'object',
        'Engagement'
    ] + [col for col in df_copy.columns if col not in ['Timestamp', 'EyeTracker-x', 'EyeTracker-y', 'Message', 'quadrant', 'AOI status', 'object', 'Engagement']]

    df_copy = df_copy[column_order]

    # Save processed data to CSV
    df_copy.to_csv(output_path, index=False)
    print("File saved to", output_path)


    from datetime import datetime

    # Define pairs of appear and disappear events to process
    event_pairs = [
        {"appear": "Appear_Distractor_FirstButterfly", "disappear": "Disappear_Distractor_FirstButterfly"},
        {"appear": "Appear_Distractor_Branch", "disappear": "Disappear_Distractor_Branch"},
        {"appear": "Appear_Distractor_SecondButterfly", "disappear": "Disappear_Distractor_SecondButterfly"},
        {"appear": "Appear_Distractor_Trunk", "disappear": "Disappear_Distractor_Trunk"},
        {"appear": "Appear_Distractor_BlueFlowers", "disappear": "Disappear_Distractor_BlueFlowers"},
        {"appear": "Appear_Target_Mushroom", "disappear": "Disappear_Target_Mushroom"}
    ]

    # Dictionary to store fixation lists and counters for each event pair
    fixation_data = {pair['appear']: [] for pair in event_pairs}
    hit_counts = {pair['appear']: 0 for pair in event_pairs}

    # Iterate over each message pair to process
    for pair in event_pairs:
        appear_event = pair["appear"]
        disappear_event = pair["disappear"]
        active = False
        e_start, e_end = None, None

        # Process messages for each event pair
        for message, timestamp in zip(df_copy['Message'], df_copy['Timestamp']):
            if isinstance(message, str):
                if appear_event in message:
                    active = True
                    e_start = int(timestamp)
                    e_start_datetime = datetime.utcfromtimestamp(e_start / 1000.0)
                elif disappear_event in message and active:
                    hit_counts[appear_event] += 1
                    active = False
                    e_end = int(timestamp)
                    e_end_datetime = datetime.utcfromtimestamp(e_end / 1000.0)
                    if e_start and e_end:
                        try:
                            duration = (e_end_datetime - e_start_datetime).total_seconds()
                            Appear_Distractor_FirstButterfly_list.append((duration, timestamp))
                            fixation_data[appear_event].append((duration, timestamp))
                        except ValueError as e:
                            print(f"Error converting timestamp: {e}")
                        e_start, e_end = None, None

    # Print fixation statistics
    print(f"\nTotal number of fixations: {total_fixations}")
    print(f"Total number of fixations inside the AoI: {fixations_in_rectangle} ({(fixations_in_rectangle / total_fixations) * 100:.2f}%)")
    print(f"Total number of fixations outside the AoI: {fixations_outside_rectangle} ({(fixations_outside_rectangle / total_fixations) * 100:.2f}%)")

    # Print fixation durations and hit counts for each event
    for pair in event_pairs:
        appear_event = pair["appear"]
        disappear_event = pair["disappear"]

        # Print the hit count for each appear event
        print(f"\nET_Hit for {appear_event}: {hit_counts[appear_event]}")

        # Print the highest duration for each appear event
        if fixation_data[appear_event]:
            max_duration = max(fixation_data[appear_event], key=lambda x: x[0])
            print(f"ET_fixation for {appear_event} Duration: {max_duration[0]} seconds")
        else:
            print(f"No {appear_event} durations recorded.")
        # Print individual fixation times
        for duration, timestamp in fixation_data[appear_event]:
            print(f"N_ET_fixation for {appear_event}: {duration} seconds")

    print("\n")
    for object_type, count in object_counts.items():
        print(f"n_et_hit for {object_type} is: {count}")

    # Count rows where EyeTracker-x and EyeTracker-y are -1
    count_et_off = ((df_copy['EyeTracker-x'] == -1) & (df_copy['EyeTracker-y'] == -1)).sum()
    print(f"\nET_Off: {count_et_off}")



    return df_copy

# Path to data and output
data_path = '/content/drive/MyDrive/ET Dataset/PT/Newdata.csv'
bg_image_path = '/content/drive/MyDrive/ET Dataset/Picture1.png'
output_path = '/content/drive/MyDrive/ET Dataset/PT/attentionData.csv'

# Run function
sustained_attention(data_path, bg_image_path, output_path)
print("-" * 180)

import pandas as pd
# File path to the CSV data
data_path = '/content/drive/MyDrive/ET Dataset/PT/attentionData.csv'
df_data = pd.read_csv(data_path, header=None)
header = pd.read_csv(data_path, nrows=1).columns.tolist()
df_data.columns = header
rows_50_to_100 = df_data.iloc[50:300]
rows_50_to_100.head(200)

















def main():
    # Cluster Raw Gaze Data to know whether eye gaze is in which AOI
    #This function takes raw eye gaze excel file of each level
    #file_path = '/content/drive/MyDrive/ET Dataset/ETdata-1st participant-3 levels.xlsm'
    #bg_image_path = '/content/drive/MyDrive/ET Dataset/Picture1.png'
    #cluster_raw_gaze_data(file_path, bg_image_path)
    #print("-" * 180)
    # Are there differences in fixation patterns between simple and complex game tasks for NDD children?
    file_path_LVL1 = '/content/drive/MyDrive/ET Dataset/PT/ET_data_neurotypical.xls'
    file_path_LVL2 = '/content/drive/MyDrive/ET Dataset/PT/ET_data_neurotypical.xls'
    file_path_LVL3 = '/content/drive/MyDrive/ET Dataset/PT/ET_data_neurotypical.xls'
    bg_image_path = '/content/drive/MyDrive/ET Dataset/Picture1.png'
    analyze_fixations(file_path_LVL1, file_path_LVL2, file_path_LVL3, bg_image_path)
    print("-" * 180)

    # Can fixation duration and location be used as indicators of cognitive load during gameplay for NDD children?
    #bg_image_path = '/content/drive/MyDrive/ET Dataset/Picture1.png'
    #csv_paths = [
     #   '/content/drive/MyDrive/ET Dataset/PT/ET_data_neurotypical.csv',
      #  '/content/drive/MyDrive/ET Dataset/PT/ET_data_neurotypical.csv',
       # '/content/drive/MyDrive/ET Dataset/PT/ET_data_neurotypical.csv'
    #]
    #plot_all_levels_forfixation_duration_analysis(bg_image_path, csv_paths)
    #print("-" * 180)
    # Extracting Reaction time, Sustained Attention and other measures
    #data_path = '/content/drive/MyDrive/ET Dataset/Test-new data.xlsx'
    #bg_image_path = '/content/drive/MyDrive/ET Dataset/Picture1.png'
    #output_path = '/content/drive/MyDrive/ET Dataset/Filled-Student10_Session34_Activity145_ATTENTION--with new timestamp.csv'
    #sustained_attention(data_path, bg_image_path, output_path)
    #print("-" * 180)
if __name__ == "__main__":
    main()

