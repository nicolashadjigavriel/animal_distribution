import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
import pandas as pd
import math
from scipy.ndimage import zoom
from matplotlib.animation import FuncAnimation

# Define parameters
simulation_time_steps = 10
frame_interval = 1
save_path = 'animal_distribution1.gif'  # Save as GIF

# Given constants
pO2 = 21.3  # Oxygen partial pressure in kPa

a = 400.17  # Scaling constant
# Constants for the calculations
b = 0.80    # Scaling exponent
T_opt = 20  # Optimal temperature (°C)
O2_opt = 21.3  # Optimal oxygen level (for normalization)
sigma = 10  # Width of Gaussian curve
C = 1       # Scaling constant for metabolic demand (optional for adjustment)

# Updated function to calculate metabolic rate R_d
def R_d(M, T, pO2):
    temp_factor = np.exp(-((T - T_opt) ** 2) / (2 * sigma ** 2))
    oxygen_factor = pO2 / O2_opt
    return a * (M ** b) * temp_factor * oxygen_factor * C

# Function to calculate daily intake I_d
def daily_intake(M, T, pO2):
    return R_d(M, T, pO2)

# Function to calculate carrying capacity K
def K(NPP, biomass_density):
    if biomass_density == 0:
        return NPP  # No competition, carrying capacity is fully available
    else:
        return NPP / biomass_density  # Decrease carrying capacity with higher biomass density

# Function to calculate survival probability
def survival_probability(K, daily_intake):
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

temperature_data = pd.read_csv('tmp_avg.csv', header=None)
T = zoom(np.nan_to_num(np.matrix.transpose(temperature_data.values), nan=999), zoom=(scale_factor, scale_factor))

# Temperature data set to constant value
#T = np.full_like(land, 20)  # Replace with a constant temperature of 20

# Filter animal data for necessary columns and drop rows with missing values
filtered_animal_data = animal_data.dropna(subset=["5-1_AdultBodyMass_g", 
                                                  "18-1_BasalMetRate_mLO2hr", 
                                                  "26-4_GR_MRLat_dd", 
                                                  "26-7_GR_MRLong_dd"])

# Extract required animal data columns
mass = filtered_animal_data["5-1_AdultBodyMass_g"]
M = mass * 0.001
latitude = filtered_animal_data["26-4_GR_MRLat_dd"]
longitude = filtered_animal_data["26-7_GR_MRLong_dd"]
R_o = filtered_animal_data["18-1_BasalMetRate_mLO2hr"]

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
    
    for individual in population:
        x, y, biomass = individual['x'], individual['y'], individual['biomass']
        local_T = T[x, y]
        R_d_value = R_d(biomass, local_T, pO2)
        I_d = daily_intake(biomass, local_T, pO2)
        local_NPP = NPP[x, y]
        
        cell_biomass_density = biomass_density[x, y]
        K_value = K(local_NPP, cell_biomass_density)
        survival_probability_value = survival_probability(K_value, I_d)
        individual['survival_probability'] = survival_probability_value

        # Movement logic with diagonal directions and boundary checking
        if random.random() < (0.05 if survival_probability_value > 0.75 else 0.1 if survival_probability_value > 0.25 else 0.25):
            possible_moves = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
            dx, dy = random.choice(possible_moves)
            new_x, new_y = (x + dx) % grid_size_lon, (y + dy) % grid_size_lat
            individual['x'], individual['y'] = new_x, new_y
            num_moves += 1

        biomass_density[x, y] += biomass
        population_density[x, y] += 1

        if survival_probability_value >= 0.2:
            surviving_population.append(individual)
            if random.random() < 0.2:
                offspring_biomass = individual['biomass']
                new_population.append({'x': x, 'y': y, 'biomass': offspring_biomass, 'age': 0})
                num_births += 1
        else:
            num_deaths += 1

    population = surviving_population + new_population

    total_biomass = sum(ind['biomass'] for ind in population)
    total_population_size = len(population)
    total_biomass_over_time.append(total_biomass)
    total_population_size_over_time.append(total_population_size)

    print(f"Time Step {t + 1}: Births = {num_births}, Deaths = {num_deaths}, Moves = {num_moves}, Biomass = {total_biomass}, Population Size = {len(population)}")
    biomass_density_plot = biomass_density*1
    biomass_density_plot[biomass_density==0] =np.nan
    plt.figure(figsize=(18, 7))
    plt.subplot(1, 3, 1)
    plt.imshow(np.transpose(land), cmap='cividis')
    plt.imshow(np.transpose(np.log10(biomass_density_plot)), cmap='inferno')
    plt.colorbar(label='Biomass Density')
    plt.title("Biomass Density")
    
    resources_plot = NPP * 1
    resources_plot[land == 0] = np.nan
    plt.subplot(1, 3, 2)
    plt.imshow(np.transpose(resources_plot), cmap='inferno')
    plt.colorbar(label='Resources')
    plt.title("Resources")
    
    population_density_plot = population_density * 1
    population_density_plot[population_density == 0] = np.nan
    plt.subplot(1, 3, 3)
    plt.imshow(np.transpose(land), cmap='cividis')
    plt.imshow(np.transpose(np.log10(population_density_plot + 1)), cmap='inferno')
    plt.colorbar(label='Population')
    plt.title("Population")
    
    plt.tight_layout()
    plt.show()
