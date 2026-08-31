import os
import pandas as pd
from floodlight.io.dfl import read_position_data_xml, read_event_data_xml, read_teamsheets_from_mat_info_xml
from databallpy.visualize import plot_events, plot_tracking_data, plot_soccer_pitch
import numpy as np
# import releavnt packages
import numpy as np; import pandas as pd; import matplotlib.pyplot as plt
from floodlight.io.dfl import read_position_data_xml, read_event_data_xml, read_teamsheets_from_mat_info_xml
import warnings
import os
from databallpy import get_game
import random
import seaborn as sns
from scipy import stats, spatial
from databallpy.visualize import plot_events, plot_tracking_data, plot_soccer_pitch
from tslearn.metrics import dtw, dtw_path
from tslearn.clustering import TimeSeriesKMeans, KShape, silhouette_score, TimeSeriesDBSCAN
from sklearn.model_selection import permutation_test_score
import pycatch22
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.covariance import MinCovDet
from scipy.stats import multivariate_normal
import gower
import scikit_posthocs as sp
import pandas


def import_match_data(game_id, path):
    """Function to import, and pre-process, a match into a DataBallPy Game object. This includes: synchronization
    of the event and position data, velocity and acceleration calculation.
    Inputs:
        game_id (str): ID of game to be imported
        path (str): local path to dataset files
    Returns:
        game (Game): Pre-processed Game object
    """
    game = get_game(tracking_data_loc= path + "DFL_04_03_positions_raw_observed_DFL-" + game_id + ".xml",
                    tracking_metadata_loc= path + "DFL_02_01_matchinformation_DFL-" + game_id + ".xml",
                    tracking_data_provider="dfl",
                    event_data_loc=path + "DFL_03_02_events_raw_DFL-" + game_id + ".xml",
                    event_metadata_loc=path + "DFL_02_01_matchinformation_DFL-" + game_id + ".xml",
                    event_data_provider="dfl")
    # synchronize event and tracking
    game.synchronise_tracking_and_event_data() 
    # add velocities and accelerations to the data
    game.tracking_data.add_velocity(column_ids=game.get_column_ids(), filter_type= "savitzky_golay", window_length=7, polyorder=2,
                                    allow_overwrite=True)
    game.tracking_data.add_acceleration(column_ids=game.get_column_ids(), filter_type= "savitzky_golay",window_length=7, polyorder=2,
                                        allow_overwrite=True)
    return game


def load_event_data(path):
    """Function to load event data for all matches stored in specified folder.
    Inputs:
        path (str): path to downloaded event data
    Outputs:
        all_events (DataFrame): DataFrame of all events across all matches
    """
    info_files = sorted([x for x in os.listdir(path) if "matchinformation" in x])
    event_files = sorted([x for x in os.listdir(path) if "events_raw" in x])
    all_events = pd.DataFrame()
    for events_file, info_file in zip(event_files, info_files):
        match_code = info_file[-10:-4]
        events, _, _ = read_event_data_xml(os.path.join(path, events_file), os.path.join(path, info_file))
        events_fullmatch = pd.DataFrame()
        for half in events:
            for team in events[half]:
                events_half = events[half][team].events
                events_half["half"] = half
                events_fullmatch = pd.concat([events_fullmatch, events_half])

        events_fullmatch["match_id"] = match_code
        all_events = pd.concat([all_events, events_fullmatch],)
    return all_events.reset_index(drop=True)


def plot_event(game, frames, passers=None, passer_traj=None):
    """Function to plot event frame(s) within game.
    Input:
        game (Game): game
        frame (int): index of frame to plot
    """
    
    x_grid, _ = np.meshgrid(np.linspace(0, 1, 15), np.linspace(0, 1, 10))
    n = len(frames)
    fig, ax = plt.subplots(1, n, figsize=(15, 3.26))
    for i in range(n):
        fig, ax[i] = plot_soccer_pitch(field_dimen=game.pitch_dimensions, pitch_color="white",
                                       
                                       fig=fig, ax=ax[i])
        fig, ax[i] = plot_tracking_data(game,
                                    frames[i],
                                    fig=fig,
                                    ax=ax[i],
                                    variable_of_interest=game.tracking_data.loc[frames[i], "frame"],
                                    )
        # plot location of passer
        if passers is not None:
            passer_num = game.player_id_to_column_id(passers[i])
            ax[i].plot(game.tracking_data.iloc[frames[i],:][passer_num+"_x"],
                                   game.tracking_data.iloc[frames[i], :][passer_num+"_y"],
                                   markerfacecolor="yellow", 
                                   marker="o", markersize=10, markeredgecolor=("green" if passer_num[:4]=="home" else "red"), zorder=3,
                                   markeredgewidth =1.5)
        if passer_traj is not None:
            ax[i].plot(passer_traj[:frames[i],0],passer_traj[:frames[i],1])

        for text in ax[i].texts:
            text.set_visible(False)


def obtain_player_position(game, team_player):
    if team_player[0]=="home":
        return game.home_players.loc[game.home_players.id==team_player[1], "position"].values[0]
    elif team_player[0]=="away":
        return game.away_players.loc[game.away_players.id==team_player[1], "position"].values[0]


def obtain_pass_trajectories(game, frames_before=75, frames_after=75, velocities=False, acc=False, centered=True):
    """Function to obtain trajectories adjacent to events.
    Inputs:
        game (Game): Game to extract trajectories
        frames_before (int): No. frames prior to event
        frames_after (int): No. frames after event
    Outputs:
        passes (DataFrame): Events corresponding to trajectories
        pass_trajectories (array): Trajectories of player adjacent to event
    """

    passes = game.pass_events.reset_index() # passes in game
    num_events = len(passes)
    pass_trajectories = np.zeros(shape=(num_events, frames_before + 1 + frames_after , 2))
    pass_velocities = np.zeros(shape=(num_events, frames_before + 1 + frames_after))
    pass_acc = np.zeros(shape=(num_events, frames_before + 1 + frames_after))
    
    # add player assigned role
    passes["role"] = passes.loc[:,["team_side", "player_id"]].apply(lambda x: obtain_player_position(game, x), axis=1)
    
    # add synched tracking frame
    passes = passes.join(game.event_data.loc[:, ["event_id", "tracking_frame", "sync_certainty"]].set_index("event_id"), on="event_id")
    
    for i in range(num_events):
        event_player = game.player_id_to_column_id(passes.loc[i,"player_id"])
        event_frame = passes.loc[i, "tracking_frame"]
        max_frame = len(game.tracking_data)-1
        pass_trajectories[i,-min(0, event_frame-frames_before):frames_before + 1 + frames_after-(max(event_frame+frames_after, max_frame)-max_frame),:] = game.tracking_data.loc[max(0, event_frame-frames_before): min(event_frame+frames_after, max_frame), [event_player+"_x", event_player+"_y"]]
        if velocities:
            pass_velocities[i,-min(0, event_frame-frames_before):frames_before + 1 + frames_after-(max(event_frame+frames_after, max_frame)-max_frame)] = game.tracking_data.loc[max(0, event_frame-frames_before): min(event_frame+frames_after, max_frame), [event_player+"_velocity"]].to_numpy().flatten()
        if acc:
            pass_acc[i,-min(0, event_frame-frames_before):frames_before + 1 + frames_after-(max(event_frame+frames_after, max_frame)-max_frame)] = game.tracking_data.loc[max(0, event_frame-frames_before): min(event_frame+frames_after, max_frame), [event_player+"_acceleration"]].to_numpy().flatten()
    if centered:
        # rotate away team trajectories
        pass_trajectories[passes["team_side"]=="away"] = pass_trajectories[passes["team_side"]=="away"] @ np.array([[-1,0],[0,-1]])
        # centre trajectories
        pass_trajectories = pass_trajectories - pass_trajectories[:,0,:][:,np.newaxis,:]
    
    if velocities and acc:
        return passes, pass_trajectories, pass_velocities, pass_acc
    elif velocities:
        return passes, pass_trajectories, pass_velocities
    elif acc:
        return passes, pass_trajectories, pass_acc
    else:
        return passes, pass_trajectories


def plot_trajectory(trajectories, title, ax=None, bounds=25):
    """Function to plot trajectories."""
    if ax is not None:
        for traj in trajectories:
            ax.plot(traj[:,0], traj[:,1], alpha=0.5, c= "tab:blue", linewidth=.3)
            ax.set_title(title)
            ax.set_xlim([-bounds, bounds])
            ax.set_ylim([-bounds, bounds])
    else:
        for traj in trajectories:
            plt.plot(traj[:,0], traj[:,1], alpha=0.5, c= "o", linewidth=.3)
        plt.title(title)    
        plt.show()


def plot_trajectory_times(game):
    """Function to plot different lengths of post-pass trajectories."""
    seconds_after = [3,5,10]
    fig, ax = plt.subplots(1, 3, figsize = (15,4))
    for ind, time in enumerate(seconds_after):
        # obtain passer trajectories for 5 seconds to pass events
        pass_events, passer_traj = obtain_pass_trajectories(game, frames_before=0,
                                                            frames_after=25*time)
        nsp_traj = passer_traj
        plot_trajectory(nsp_traj, f"{time} Seconds", ax[ind])
        ax[ind].grid(alpha=0.5)
        ax[ind].set_xlabel("Longitudinal Displacement (m)")
        ax[ind].set_ylabel("Lateral Displacement (m)")
    plt.show()


def plot_final_location(trajectories, role, ax=None, bounds=20):
    """Function to plot denisty plot of trajectory end locations."""
    if ax is not None:
        ax.set_title(role)
        ax.set_xlim([-bounds, bounds])
        ax.set_ylim([-bounds, bounds])
        sns.kdeplot(x=trajectories[:,-1,0], y=trajectories[:,-1,1], cmap="Blues", fill=True, ax=ax, shade=True)
        ax.axvline(x=0, ymin=-bounds, ymax=bounds, linewidth=.7, linestyle="--")


def get_distance(game, player_id, frame, frames_before, frames_after):
    """Function to extract distance covered by player after a pass."""
    max_frame = len(game.tracking_data)-1
    dist = game.tracking_data.get_covered_distance(column_ids = [game.player_id_to_column_id(player_id)],
                                                    start_idx = frame-frames_before, end_idx = min(frame+frames_after,max_frame)).iloc[0,0]
    return dist


def plot_distance_covered_dist(game, pass_events, roles):
    """Function for plotting post pass distance covered."""
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    for i in [1, 3, 5]:
        post_pass_dist = pass_events[["player_id", "tracking_frame"]].apply(lambda x: get_distance(game, x["player_id"], x["tracking_frame"], 0, 25*i), axis=1)
        ax[0].hist(post_pass_dist, alpha=0.5, density=True, label = f"{i} Seconds")
    ax[0].legend(title="Post-Pass Time")
    ax[0].set_title("Post-Pass Distance Covered")
    for role in roles:
        ax[1].hist(post_pass_dist[pass_events["role"]==role], alpha=0.5, density=True, label = role, bins=15)
    ax[1].legend(title="Post-Pass Time")
    ax[1].set_title("Post-Pass Distance Covered")
    plt.show()


def compare_synchronization(game):
    """Function to visually compare synchronization methods."""
    # sample 3 random passes
    np.random.seed(456)
    passes = game.event_data[game.event_data.databallpy_event=="pass"]
    passes_sample = np.random.choice(passes["event_id"], 3)
    passers = passes.loc[passes["event_id"].isin(passes_sample),"player_id"].to_list()
    
    # plot the positions at the frame closest to these timestamps
    print("Kick-off Lag Corrected Synchronization:")
    kickoff_bias = passes.loc[0,"datetime"] - game.tracking_data.loc[0,"datetime"]
    closest_frames = [np.argmin(abs(time - kickoff_bias - game.tracking_data["datetime"]), axis=0) for time in passes.loc[passes["event_id"].isin(passes_sample),"datetime"]]
    plot_event(game, closest_frames, passers=passers); plt.show()

    # plot the position at the synchronized frame
    print("DataBallPy Synchronization:")
    plot_event(game, passes.loc[passes["event_id"].isin(passes_sample),"tracking_frame"].to_list(), passers=passers)
    plt.show()


def pass_interval_times(game):
    """Function to return the interval times between passes in a given game, excluding stoppages prior to set-piece events."""
    pass_events, passer_traj = obtain_pass_trajectories(game, frames_before=0,
                                                        frames_after=25*5)
    # distribution of time between passes
    h1_pass = pass_events[pass_events["period_id"]==1] # first half passes
    h2_pass = pass_events[pass_events["period_id"]==2] # second half passes

    # waiting times between passes
    h1_pass["p_wait"] = (h1_pass.sort_values("tracking_frame")["tracking_frame"] -
                        h1_pass.sort_values("tracking_frame")["tracking_frame"].shift(1))
    h2_pass["p_wait"] = (h2_pass.sort_values("tracking_frame")["tracking_frame"] -
                        h2_pass.sort_values("tracking_frame")["tracking_frame"].shift(1))

    # remove waiting time when 2nd pass is set piece
    h1 = h1_pass[h1_pass["set_piece"] == "no_set_piece"]["p_wait"]
    h2 = h2_pass[h2_pass["set_piece"] == "no_set_piece"]["p_wait"]

    # convert to time (seconds)
    time_between_passes = pd.concat([h1, h2]).dropna().values/25
    return time_between_passes


def plot_post_set_piece_movements(game):
    """Function to plot post-pass trajectories by whether pass is set-piece or open play."""
    fig, ax = plt.subplots(1, 2, figsize = (8,3))
    pass_events, passer_traj = obtain_pass_trajectories(game, frames_before=0,
                                                        frames_after=25*3)
    nsp_traj = passer_traj[(pass_events["set_piece"]=="no_set_piece")]
    sp_traj = passer_traj[(pass_events["set_piece"]!="no_set_piece")]
    plot_final_location(nsp_traj, f"No Set Piece", ax[0],bounds=15)
    plot_final_location(sp_traj, f"Set Piece", ax[1], bounds=15)
    plt.show()


def plot_trajectories_by_role(game, roles):
    """Function to plot post-pass trajectories by passer role."""
    pass_events, passer_traj = obtain_pass_trajectories(game, frames_before=0, frames_after=25*3)
    fig, ax = plt.subplots(2, 4, figsize = (15,8))
    for ind, role in enumerate(roles):
        role_trajectory = passer_traj[((pass_events["role"]==role) &
                                    (pass_events["set_piece"]=="no_set_piece"))]
        plot_trajectory(role_trajectory, role, ax[ind//2, ind % 2], bounds=15)
    # plot density plot of final locations
    for ind, role in enumerate(roles):
        role_trajectory = passer_traj[(pass_events["role"]==role) & (pass_events["set_piece"]=="no_set_piece")]
        plot_final_location(role_trajectory, role, ax[ind//2, 2+ind % 2], bounds=15)
    plt.show()
    

def plot_trajectories_by_pitch_loc(pass_events, passer_traj, thirds_l):
    """Function to plot post-pass trajectories by pass location on the pitch."""
    # plot trajectories by location of the pass on the pitch
    fig, ax = plt.subplots(1, 3, figsize = (10,3))
    for ind, location in enumerate(np.sort(pass_events["location"].unique().dropna())):
        loc_trajectory = passer_traj[((pass_events["location"]==location) &
                                    (pass_events["set_piece"]=="no_set_piece"))]
        plot_final_location(loc_trajectory, thirds_l[ind], ax[ind], bounds=15)
    plt.show()



def plot_velocity_trajectories(pass_velocities, pass_events, interval, roles):
    """Function to plot average post-pass velocity trajectories, including by passer role."""
    fig, ax = plt.subplots(1,2, figsize = (10,3))
    # normalize the velocity time series
    norm_pass_velocities = pass_velocities/np.max(pass_velocities,axis=1).reshape((728,1))
    median_velocity = np.nanmedian(norm_pass_velocities,axis=0)/ np.max(np.nanmedian(norm_pass_velocities,axis=0))
    mean_velocity = np.nanmean(norm_pass_velocities,axis=0)/ np.max(np.nanmean(norm_pass_velocities,axis=0))
    # plot average velocities
    ax[0].plot(median_velocity, label="Median"); ax[0].plot(mean_velocity, label="Mean")
    ax[0].legend(); ax[0].vlines(interval,0,1, colors="red", linestyles="dashed")
    ax[0].set_title("All Passes")
    for ind, role in enumerate(roles):
        role_velocities= norm_pass_velocities[(pass_events["role"]==role) & (pass_events["set_piece"]=="no_set_piece")]
        ax[1].plot(np.nanmedian(role_velocities,axis=0)/np.nanmax(np.nanmedian(role_velocities,axis=0)), label=role)
    ax[1].vlines(75,0,1, colors="red", linestyles="dashed")
    ax[1].legend(); ax[1].set_title("Player Role")
    plt.show()


def simpl_event(x):
    if ("_Play" in x) & ("Pass" in x):
        return "Set Piece Pass"
    elif ("_Play" in x) & ("Cross" in x):
        return "Set Piece Cross"
    elif ("Cross" in x):
        return "Cross"
    else:
        return "Pass"


def plot_trajectories_goalkeeper(pass_event_consol):
    """Function to plot post-pass trajectories final location for outfield player vs goalkeeper."""
    # plot density plot for goalkeeper vs non-goalkeeper
    fig, ax = plt.subplots(1,2, figsize =(12,4))
    x_bounds = 30
    y_bound = 20
    # plot for goalkeeper
    gk_fl = pass_event_consol[pass_event_consol["role"]=="goalkeeper"]["post_pass_5_fl"]
    ax[0].set_title("Goalkeeper")
    ax[0].set_xlim([-x_bounds, x_bounds])
    ax[0].set_ylim([-y_bound, y_bound])
    ax[0].set_aspect('equal')
    sns.kdeplot(x=gk_fl.apply(lambda x: x[0]), y=gk_fl.apply(lambda x: x[1]), cmap="Blues", fill=True, ax=ax[0], shade=True)
    ax[0].set_xlabel("Longidudinal Displacement (m)")
    ax[0].set_ylabel("Lateral Displacement (m)")
    ax[0].grid(alpha=0.25)
    # plot for non goalkeeper
    ngk_fl = pass_event_consol[pass_event_consol["role"]!="goalkeeper"]["post_pass_5_fl"]
    ax[1].set_title("Outfield")
    ax[1].set_xlim([-x_bounds, x_bounds])
    ax[1].set_ylim([-y_bound, y_bound])
    ax[1].set_aspect('equal')
    ax[1].set_xlabel("Longidudinal Displacement (m)")
    sns.kdeplot(x=ngk_fl.apply(lambda x: x[0]), y=ngk_fl.apply(lambda x: x[1]), cmap="Blues", fill=True, ax=ax[1], shade=True)
    ax[1].set_ylabel("Lateral Displacement (m)")
    ax[1].grid(alpha=0.25)
    plt.show()