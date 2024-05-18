import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

def read_excel_file(file_path):
    """
    Read an Excel file using Pandas.

    Parameters:
    file_path : str
        Path to the Excel file.

    Returns:
    df : pandas DataFrame
        DataFrame containing the data read from the Excel file.
    """
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        print("An error occurred while reading the Excel file:", e)
        return None

def coupling_angle_cyclogram(proximal_joint_angle, distal_joint_angle, epsilon):
    """
    Compute coupling angle in degrees between proximal and distal joint angles.
    Based on the article:
    "Analysing patterns of coordination and patterns of control using novel data
    visualisation techniques in vector coding Robert A. Needham,*, Roozbeh Naemi,
    Joseph Hamill, Nachiappan Chockalingam, The Foot 44 (2020) 101678 "

    Parameters:
    proximal_joint_angle : numpy array
        Proximal joint angle in degrees.
    distal_joint_angle : numpy array
        Distal joint angle in degrees.
    epsilon : float
        Tolerance level.

    Returns:
    coupling_angle : numpy array
        Coupling angle in degrees between proximal and distal joint angle.

    """
    proximal_joint_angle_diff = np.diff(proximal_joint_angle)
    distal_joint_angle_diff = np.diff(distal_joint_angle)

    coupling_angles = np.zeros(proximal_joint_angle_diff.shape[0])
    x_coupling_angles = np.zeros(proximal_joint_angle_diff.shape[0])
    y_coupling_angles = np.zeros(proximal_joint_angle_diff.shape[0])

    for idx in range(proximal_joint_angle_diff.shape[0]):
        if proximal_joint_angle_diff[idx] > 0:
            coupling_angle = np.arctan(distal_joint_angle_diff[idx] / proximal_joint_angle_diff[idx]) * 180/np.pi
        elif proximal_joint_angle_diff[idx] < 0:
            coupling_angle = np.arctan(distal_joint_angle_diff[idx] / proximal_joint_angle_diff[idx]) * 180/np.pi + 180 / np.pi
        elif proximal_joint_angle_diff[idx] < epsilon and distal_joint_angle_diff[idx] > 0:
            coupling_angle = 90
        elif proximal_joint_angle_diff[idx] < epsilon and distal_joint_angle_diff[idx] < 0:
            coupling_angle = -90
        elif proximal_joint_angle_diff[idx] < 0 and distal_joint_angle_diff[idx] <  epsilon:
            coupling_angle = -180
        elif proximal_joint_angle_diff[idx] < epsilon and distal_joint_angle_diff[idx] < epsilon:
            coupling_angle = np.nan

        if coupling_angle < 0:
            coupling_angle = coupling_angle + 360

        coupling_angles[idx] = coupling_angle
        x_coupling_angles[idx] = np.cos(coupling_angle)
        y_coupling_angles[idx] = np.sin(coupling_angle)

    return {'coupling_angles': coupling_angles, 'x_coupling_angles': x_coupling_angles, 'y_coupling_angles': y_coupling_angles}

def mean_coupling_angle_metrics(coupling_angles_x_grouped, coupling_angles_y_grouped, epsilon):
    """
    Calculate coupling metrics.

    Parameters:
    coupling_angles_x_grouped : numpy array
                                Grouped coupling angles for x.
    coupling_angles_y_grouped : numpy array
                                Grouped coupling angles for y.

    Returns:
    coupling_angle_bar :            numpy array
                                    Mean coupling angle in degrees.
    r_bar :                         numpy array
                                    Magnitude of the mean vector.
    coupling_angle_variability :    numpy array
                                    Variability of the coupling angles in degrees.
    """
    coupling_angle_bar = np.zeros(coupling_angles_x_grouped.shape[0])
    r_bar = np.zeros(coupling_angles_x_grouped.shape[0])
    coupling_angle_variability = np.zeros(coupling_angles_x_grouped.shape[0])

    for idx in range(0, coupling_angles_x_grouped.shape[0]):
        x_bar = np.mean(coupling_angles_x_grouped[idx, :])
        y_bar = np.mean(coupling_angles_y_grouped[idx, :])

        if x_bar > 0 and y_bar > 0:
            angle = np.arctan(y_bar / x_bar) * 180 / np.pi
        elif x_bar < 0:
            angle = np.arctan(y_bar / x_bar) * 180 / np.pi + 180
        elif x_bar > 0 and y_bar < 0:
            angle = np.arctan(y_bar / x_bar) * 180 / np.pi + 360

        elif x_bar <  epsilon and y_bar > 0:
            angle = 90
        elif x_bar <  epsilon and y_bar < 0:
            angle = -90
        elif x_bar <  epsilon and y_bar <  epsilon:
            angle = np.nan

        coupling_angle_bar[idx] = angle

        ri = (x_bar ** 2 + y_bar ** 2) ** 0.5
        r_bar[idx] = ri

        coupling_angle_variability[idx] = (2 * (1 - ri)) ** 0.5 * 180 / np.pi

    return {'coupling_angle_bar': coupling_angle_bar,
            'r_bar':r_bar,
            'coupling_angle_variability': coupling_angle_variability}

if __name__ == "__main__":
    # Importing necessary modules
    import doctest

    # Define input folder and file name
    input_folder = r'C:\Users\admin\Downloads'
    fname = r'shots_1st_ankurS10.xlsx'
    fpath = os.path.join(input_folder, fname)

    # Read Excel file
    fdata = read_excel_file(fpath)
    fdata_header = fdata.columns

    # Extract proximal and distal joint angle data from the Excel file
    proximal_joint = fdata['RightHipFlexion_Extension']
    distal_joint = fdata['RightKneeFlexion_Extension']

    # Calculate coupling angles for three repetitions
    coupling_angle_rep1 = coupling_angle_cyclogram(proximal_joint, distal_joint, 0.001)
    coupling_angle_rep2 = coupling_angle_cyclogram(proximal_joint, distal_joint, 0.001)
    coupling_angle_rep3 = coupling_angle_cyclogram(proximal_joint, distal_joint, 0.001)

    # Group coupling angles for x and y
    coupling_angles_x_grouped = np.vstack((coupling_angle_rep1['x_coupling_angles'],
                                            coupling_angle_rep2['x_coupling_angles'],
                                            coupling_angle_rep3['x_coupling_angles'])).T

    coupling_angles_y_grouped = np.vstack((coupling_angle_rep1['y_coupling_angles'],
                                            coupling_angle_rep2['y_coupling_angles'],
                                            coupling_angle_rep3['y_coupling_angles'])).T

    # Calculate mean coupling angles and other metrics
    mean_coupling_angles_data = mean_coupling_angle_metrics(coupling_angles_x_grouped, coupling_angles_y_grouped, 0.0001)

    # Plot joint angle data
    plt.plot(proximal_joint, distal_joint)
    plt.xlabel('Proximal joint [º]')
    plt.ylabel('Distal joint [º]')
    plt.legend(['rep1'])

    # Time series start
    plt.scatter(proximal_joint.iloc[0], distal_joint.iloc[0], s=50, color='green')
    plt.xlim()
    plt.show()

    # Plot coupling angles for each repetition and mean coupling angle
    plt.plot(coupling_angle_rep1['coupling_angles'])
    plt.plot(coupling_angle_rep2['coupling_angles'])
    plt.plot(coupling_angle_rep3['coupling_angles'])
    plt.plot(mean_coupling_angles_data['coupling_angle_bar'])
    plt.legend(['rep1', 'rep2', 'rep3', 'mean_coupling_angle'])
    plt.xlabel('Degrees [º]')
    plt.ylabel('Frames [n]')
    plt.xlim([0, coupling_angle_rep1['coupling_angles'].shape[0]])
    plt.ylim([0, 360])
    plt.show()

    # Plot mean coupling angle variability
    plt.plot(mean_coupling_angles_data['coupling_angle_variability'])
    plt.xlim([0, coupling_angle_rep1['coupling_angles'].shape[0]])
    plt.xlabel('Frames [n]')
    plt.ylabel('Degrees [º]')
    plt.legend(['mean_coupling_angle_variability'])
    plt.show()
