# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:47:02 2024

@author: nicol
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 11:28:05 2024

@author: User
"""


import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
import pandas as pd
import math
from scipy.ndimage import zoom
from matplotlib.animation import FuncAnimation


# Define parameters
simulation_time_steps = 100

frame_interval = 1
save_path = 'animal_distirbution1.gif'  # Save as GIF



# Given constants
pO2 = 21.3  # Oxygen partial pressure in kPa
alpha = 0  # Exponent for oxygen in metabolic rate equation
beta = 0  # Temperature coefficient
c = 1

# Function to calculate metabolic rate R_d
def R_d(M, T, pO2, alpha, beta):
    return 1.5 * (M**(0.75)) * (pO2**alpha) * np.exp(beta * T)

# Function to calculate daily intake I_d
def daily_intake(M, T, pO2, alpha, beta):
    r_d = R_d(M, T, pO2, alpha, beta)
    return r_d

def K(NPP, biomass_density):
    if biomass_density == 0:
        return NPP  # No competition, carrying capacity is fully available
    else:
        return NPP / biomass_density  # Decrease carrying capacity with higher biomass density

def survival_probability(K, daily_intake):
    #return 1 / (1 + np.exp(-(K -daily_intake)))
    return min(1, max(0, K / (K + daily_intake)))  # Smoothing the curve a bit

scale_factor = 0.1

# Load and process data
animal_data_file = "pantheria_filtered_data_for_dist_plots.csv"
animal_data = pd.read_csv(animal_data_file)

land_values_lowres_df = pd.read_csv('land_cru.csv', header=None)
land = zoom(np.matrix.transpose(land_values_lowres_df.values), zoom=(scale_factor, scale_factor))

resource_values_lowres_df = pd.read_csv('NPP.csv', header=None)
resource_transpose = np.matrix.transpose(resource_values_lowres_df.values)
resource_scaled = 10 * np.divide(resource_transpose, np.nanmax(resource_transpose))
resource_scaled = np.nan_to_num(resource_scaled, nan=0)
NPP = zoom(resource_scaled * 100, zoom=(scale_factor, scale_factor))

lat_df = pd.read_csv('lat_cru.csv', header=None)
lat = zoom(np.squeeze(lat_df.values), zoom=(scale_factor))

lon_df = pd.read_csv('lon_cru.csv', header=None)
lon = zoom(np.squeeze(lon_df.values), zoom=(scale_factor))

# temperature_data = pd.read_csv('tmp_avg.csv', header=None)
# T = zoom(np.nan_to_num(np.matrix.transpose(temperature_data.values), nan=999), zoom=(scale_factor, scale_factor))

# Temperature data set to constant value
T = np.full_like(land, 1)  # Replace with a constant temperature of 20

# Print dimensions to verify data shapes
print("Temperature dimensions:", T.shape)
print("NPP (Resources) dimensions:", NPP.shape)
print("Land dimensions:", land.shape)

# Filter animal data for necessary columns and drop rows with missing values
filtered_animal_data = animal_data.dropna(subset=["5-1_AdultBodyMass_g", 
                                                  "18-1_BasalMetRate_mLO2hr", 
                                                  "26-4_GR_MRLat_dd", 
                                                  "26-7_GR_MRLong_dd"])

# Extract required animal data columns
mass = filtered_animal_data["5-1_AdultBodyMass_g"]
M = mass*0.001
latitude = filtered_animal_data["26-4_GR_MRLat_dd"]
longitude = filtered_animal_data["26-7_GR_MRLong_dd"]
R_o = filtered_animal_data["18-1_BasalMetRate_mLO2hr"]



# # Extract required animal data columns
# mass = filtered_animal_data["5-1_AdultBodyMass_g"]

# # Filter data for animals with M > 1 kg
# filtered_animal_data = filtered_animal_data[mass > 1000].copy()  # mass > 1000 grams (1 kg)

# # Recalculate necessary variables after filtering
# mass = filtered_animal_data["5-1_AdultBodyMass_g"]
# M = mass * 0.001
# latitude = filtered_animal_data["26-4_GR_MRLat_dd"]
# longitude = filtered_animal_data["26-7_GR_MRLong_dd"]
# R_o = filtered_animal_data["18-1_BasalMetRate_mLO2hr"]

M_max = np.max(M)
M_min = np.min(M)




grid_size_lon, grid_size_lat = land.shape

# Initialize population
population = []
for _ in range(500):
    while True:
        x = np.random.randint(0, grid_size_lon)
        y = np.random.randint(0, grid_size_lat)
        if land[x, y] == 1 and NPP[x, y] > 0:
            break
    population.append({'x': x, 'y': y, 'biomass': np.random.uniform(1, M_max), 'age': 0})

# Prepare figure and animation function
fig, axs = plt.subplots(1, 4, figsize=(40, 20))
time_text = fig.text(0.5, 0.95, '', ha='center', va='center', fontsize=16, color='blue')

total_biomass_over_time = []
total_population_size_over_time = []



# Simulation loop
for t in range(simulation_time_steps):
    new_population = []  # To store new offspring
    surviving_population = []  # To store individuals that survive
    num_births = 0
    num_deaths = 0
    num_moves = 0
    
    # Reset biomass and population density for the current time step
    biomass_density = np.zeros((grid_size_lon, grid_size_lat))
    population_density = np.zeros((grid_size_lon, grid_size_lat))
    
    # Reset survival probability distribution for the current time step
    survival_prob_distribution = np.zeros((grid_size_lon, grid_size_lat))
    
    for individual in population:
        x, y, biomass = individual['x'], individual['y'], individual['biomass']
        local_T = T[x, y]
        R_d_value = R_d(biomass, local_T, pO2, alpha, beta)
        I_d = daily_intake(biomass, local_T, pO2, alpha, beta)
        local_NPP = NPP[x, y]
        
        # Calculate the total biomass in the current cell
        cell_biomass_density = biomass_density[x, y]
        
        # Update carrying capacity considering biomass density
        K_value = K(local_NPP, cell_biomass_density)
        
        # Update survival probability based on new K function
        survival_probability_value = survival_probability(K_value, I_d)
        individual['survival_probability'] = survival_probability_value

        # Save survival probability in the matrix
        #survival_prob_distribution[x, y] += survival_probability_value

        # Default new_x and new_y to the current position
        new_x, new_y = x, y

        # Movement logic with boundary checking
        # Movement logic with diagonal directions and boundary checking
        #if random.random() < (0.05 if survival_probability_value > 0.75 else 0.1 if survival_probability_value > 0.25 else 0.25):
        # if random.random() < survival_probability_value:
        #     # List of all 8 possible moves: vertical, horizontal, and diagonal
        #     possible_moves = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        #     dx, dy = random.choice(possible_moves)
        #     new_x, new_y = (x + dx) % grid_size_lon, (y + dy) % grid_size_lat  # Ensure boundary wrap
        #     individual['x'], individual['y'] = new_x, new_y
        #     num_moves += 1
            
        if random.random() < (0.05 if survival_probability_value > 0.75 else 0.1 if survival_probability_value > 0.25 else 0.25):
            # List of all 8 possible moves: vertical, horizontal, and diagonal
            possible_moves = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
            dx, dy = random.choice(possible_moves)
            new_x, new_y = (x + dx) % grid_size_lon, (y + dy) % grid_size_lat  # Ensure boundary wrap
            individual['x'], individual['y'] = new_x, new_y
            num_moves += 1
            
            


        # Calculate new survival after movement
        individual['R_d'] = R_d(biomass, T[new_x, new_y], pO2, alpha, beta)
        I_d = daily_intake(biomass, T[new_x, new_y], pO2, alpha, beta)
        cell_biomass_density = biomass_density[new_x, new_y]
        
        # Recalculate carrying capacity and survival probability after movement
        K_value = K(NPP[new_x, new_y], cell_biomass_density)
        individual['survival_probability'] = survival_probability(K_value, I_d)

        # Handle death
        # Handle death
        if individual['survival_probability'] < 0.2:
            num_deaths += 1
            # Remove the individual from the grid (decrease biomass and population density)
            biomass_density[x, y] -= biomass
            population_density[x, y] -= 1
            # Do not add the individual to the surviving_population (this removes them from the population)
        else:
            # Individual survives, add to surviving_population
            surviving_population.append(individual)

    
                # Handle reproduction
            if random.random() < 0.2:  # 20% chance of reproduction
                offspring_biomass = individual['biomass']   # Offspring biomass as % of parent's biomass
                new_population.append({'x': new_x, 'y': new_y, 'biomass': offspring_biomass, 'age': 0})
                num_births += 1

        # Update biomass and population density in grid cell
        biomass_density[new_x, new_y] += biomass
        population_density[new_x, new_y] += 1
        
    # for i in range(grid_size_lon):
    #     for j in range(grid_size_lat):
    #         if population_density[i, j] > 0:
    #             survival_prob_distribution[i, j] /= population_density[i, j]


    # Update the population with surviving individuals and new offspring
    population = surviving_population + new_population

    ## Record total biomass and population size at each timestep
    total_biomass = sum(ind['biomass'] for ind in population)
    total_population_size = len(population)
    
    # Append the metrics to their respective lists
    total_biomass_over_time.append(total_biomass)
    total_population_size_over_time.append(total_population_size) 

    # Print step info
    print(f"Time Step {t + 1}: Births = {num_births}, Deaths = {num_deaths}, Moves = {num_moves}, Biomass = {total_biomass}, Population Size = {len(population)}")
    
    bmin_value = np.min(biomass_density)
    bmax_value = np.max(biomass_density)
    #print(f"Min biomass density: {bmin_value}, Max biomass density: {bmax_value}")
    pmin_value = np.min(population_density)
    pmax_value = np.max(population_density)
    # Plotting each step
    biomass_density_plot = biomass_density*1
    biomass_density_plot[biomass_density==0] =np.nan
    
plt.figure(figsize=(18, 7))
plt.subplot(1, 3, 1)
plt.imshow(np.transpose(land), cmap='cividis')
plt.imshow(np.transpose(np.log10(biomass_density_plot)), cmap='inferno')
plt.colorbar(label='Biomass Density')
plt.title("Biomass Density")



resources_plot =  NPP*1
resources_plot[land==0] =np.nan
plt.subplot(1, 3, 2)
plt.imshow(np.transpose(resources_plot), cmap='inferno')
plt.colorbar(label='Resources')
plt.title("Resources")

# plt.subplot(1, 5, 3)
# plt.imshow(np.transpose(land), cmap='cividis')
# plt.colorbar(label='Land')
# plt.title("Land")

population_density_plot =  population_density*1
population_density_plot[population_density==0] =np.nan

plt.subplot(1, 3, 3)
plt.imshow(np.transpose(land), cmap='cividis')
plt.imshow(np.transpose(np.log10(population_density_plot)), cmap='inferno')#, vmin= 0, vmax=200)
plt.colorbar(label='Population')
plt.title("Population")

plt.tight_layout()
plt.show()




# Load animal distribution data
animal_distribution_df = pd.read_csv('actual_animal_density_distribution.csv')

# Filter rows with missing latitude, longitude, or biomass density
animal_distribution_df = animal_distribution_df.dropna(subset=['Lat', 'Lon', 'Biomass_density'])


# Define the grid resolution (36x72)
lat = np.linspace(-90, 90, 36)  # Latitude grid (36 points)
lon = np.linspace(-180, 180, 72)  # Longitude grid (72 points)

# Initialize the heatmap grid
heatmap = np.zeros((len(lat), len(lon)), dtype=np.float32)

# Map biomass density to grid cells
for _, row in animal_distribution_df.iterrows():
    lat_val = row['Lat']
    lon_val = row['Lon']
    actual_biomass_density = row['Biomass_density']

    # Ensure data aligns within the grid boundaries
    if lat_val < lat.min() or lat_val > lat.max() or lon_val < lon.min() or lon_val > lon.max():
        continue

    # Find the closest grid indices for latitude and longitude
    lat_idx = (np.abs(lat - lat_val)).argmin()
    lon_idx = (np.abs(lon - lon_val)).argmin()

    # Accumulate biomass density into the grid cell
    heatmap[lat_idx, lon_idx] += actual_biomass_density

# Mask cells with zero biomass density for better visualization
heatmap[heatmap == 0] = np.nan

# Define a custom colormap
colors = [
    (0.0, "blue"),    # Close to 0
    (0.1, "green"),   # Around 10
    (0.3, "yellow"),  # Around 100
    (0.6, "orange"),  # Around 1000
    (1.0, "darkred")  # Around 10000
]
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

# Plot the corrected heatmap
plt.figure(figsize=(10, 8))
plt.imshow(
    heatmap,
    origin='lower',
    cmap=custom_cmap,  # Use the custom colormap
    norm=LogNorm(vmin=1, vmax=10000),  # Logarithmic normalization
    extent=[lon.min(), lon.max(), lat.min(), lat.max()]  # Ensure correct geographic scaling
)
cbar = plt.colorbar()
cbar.set_label('Herbivore biomass (kg/km²)')
plt.title('Grid-Cell-Based Heatmap of Biomass Density (36x72 Resolution)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()


# Plot the corrected heatmap
plt.figure(figsize=(10, 8))
plt.imshow(
    np.rot90(biomass_density_plot),
    origin='lower',
    cmap=custom_cmap,  # Use the custom colormap
    norm=LogNorm(vmin=1, vmax=10000),  # Logarithmic normalization
    extent=[lon.min(), lon.max(), lat.min(), lat.max()]  # Ensure correct geographic scaling
)
cbar = plt.colorbar()
cbar.set_label('Modeled biomass (kg/km²)')
plt.title('Grid-Cell-Based Heatmap of Biomass Density (36x72 Resolution)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()



# Additional Plots: Biomass Density vs. Latitude and Longitude
# Aggregate biomass density for each unique latitude and longitude
lat_bins = np.linspace(lat.min(), lat.max(), len(lat))
lon_bins = np.linspace(lon.min(), lon.max(), len(lon))

biomass_vs_lat = np.nansum(heatmap, axis=1)  # Sum along longitude (columns)
biomass_vs_lon = np.nansum(heatmap, axis=0)  # Sum along latitude (rows)

# Plot Biomass Density vs Latitude
plt.figure(figsize=(10, 6))
plt.plot(lat_bins, biomass_vs_lat, label='Biomass Density vs Latitude')
plt.title('Biomass Density vs Latitude')
plt.xlabel('Latitude')
plt.ylabel('Summed Biomass Density (kg/km²)')
plt.grid(True)
plt.legend()
plt.show()

# Plot Biomass Density vs Longitude
plt.figure(figsize=(10, 6))
plt.plot(lon_bins, biomass_vs_lon, label='Biomass Density vs Longitude', color='orange')
plt.title('Biomass Density vs Longitude')
plt.xlabel('Longitude')
plt.ylabel('Summed Biomass Density (kg/km²)')
plt.grid(True)
plt.legend()
plt.show()


# Extract actual biomass density (grouping by latitude and longitude)
actual_biomass_density_lat = biomass_vs_lat
actual_biomass_density_lon = biomass_vs_lon

# Aggregate modeled biomass density by latitude and longitude
lat_bins = np.linspace(np.min(lat), np.max(lat), biomass_density.shape[1])
lon_bins = np.linspace(np.min(lon), np.max(lon), biomass_density.shape[0])

# Summing modeled biomass density along latitude and longitude
modeled_biomass_density_lat = np.sum(biomass_density, axis=0)
modeled_biomass_density_lon = np.sum(biomass_density, axis=1)

# Plot biomass density against latitude
plt.figure(figsize=(12, 6))
plt.plot(lat_bins, modeled_biomass_density_lat, label="Modeled Biomass Density", color="blue", lw=2)
plt.plot(
    lat_bins,  # Use lat_bins here, as it corresponds to latitude values
    actual_biomass_density_lat,
    label="Actual Biomass Density",
    color="orange",
    lw=2,
    linestyle="--",
)

plt.xlabel("Latitude")
plt.ylabel("Biomass Density")
plt.title("Biomass Density vs. Latitude")
plt.legend()
plt.grid(True)
plt.show()

# Plot biomass density against longitude
plt.figure(figsize=(12, 6))
plt.plot(lon_bins, modeled_biomass_density_lon, label="Modeled Biomass Density", color="blue", lw=2)
plt.plot(
    lon_bins,  # Use lat_bins here, as it corresponds to latitude values
    actual_biomass_density_lon,
    label="Actual Biomass Density",
    color="orange",
    lw=2,
    linestyle="--",
)

plt.xlabel("Longitude")
plt.ylabel("Biomass Density")
plt.title("Biomass Density vs. Longitude")
plt.legend()
plt.grid(True)
plt.show()