
import numpy as np
import random
import matplotlib.pyplot as plt
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
    return 1.5 * (M**(2/3)) * (pO2**alpha) * np.exp(beta * T)

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
    return 1 / (1 + np.exp(-(K -daily_intake)))
   #return min(1, max(0, K / (K + daily_intake)))  # Smoothing the curve a bit

scale_factor = 0.1

# Load and process data
animal_data_file = "pantheria_filtered_data_for_distirbution_plots.csv"
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
T = np.full_like(land, 20)  # Replace with a constant temperature of 20

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

M_max = np.max(M)
M_min = np.min(M)




grid_size_lon, grid_size_lat = land.shape

# Initialize population
population = []
for _ in range(len(M)):
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
        survival_prob_distribution[x, y] += survival_probability_value

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
        
    for i in range(grid_size_lon):
        for j in range(grid_size_lat):
            if population_density[i, j] > 0:
                survival_prob_distribution[i, j] /= population_density[i, j]


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
plt.subplot(1, 4, 1)
plt.imshow(np.transpose(land), cmap='cividis')
plt.imshow(np.transpose(np.log10(biomass_density_plot)), cmap='inferno')
plt.colorbar(label='Biomass Density')
plt.title("Biomass Density")



resources_plot =  NPP*1
resources_plot[land==0] =np.nan
plt.subplot(1, 4, 2)
plt.imshow(np.transpose(resources_plot), cmap='inferno')
plt.colorbar(label='Resources')
plt.title("Resources")

# plt.subplot(1, 5, 3)
# plt.imshow(np.transpose(land), cmap='cividis')
# plt.colorbar(label='Land')
# plt.title("Land")

population_density_plot =  population_density*1
population_density_plot[population_density==0] =np.nan

plt.subplot(1, 4, 3)
plt.imshow(np.transpose(land), cmap='cividis')
plt.imshow(np.transpose(np.log10(population_density_plot)), cmap='inferno')#, vmin= 0, vmax=200)
plt.colorbar(label='Population')
plt.title("Population")

# Plot Survival Probability Distribution
survival_prob_plot = survival_prob_distribution 
np.maximum(population_density, 1)
survival_prob_plot[population_density == 0] = np.nan
# Normalize by population density
plt.subplot(1, 4, 4)
plt.imshow(np.transpose(land), cmap='cividis')
plt.imshow(np.transpose(survival_prob_plot),cmap='inferno', vmin= 0, vmax=1)
plt.colorbar(label='Survival Probability')
plt.title("Survival Probability Distribution")

plt.tight_layout()
plt.show()

# Initialize figure for environmental layers
plt.figure(figsize=(18, 7))

# Setting the extent based on global latitude/longitude ranges for proper alignment
extent = [-180, 180, -90, 90]

# Plot 1: Resources (NPP) overlaying land
resources_plot = NPP*1
resources_plot[land == 0] = np.nan  # Masking water bodies
plt.subplot(1, 5, 1)
plt.imshow(np.transpose(resources_plot), cmap='inferno', extent=extent)
plt.colorbar(label='Resources')
plt.title("Resources")

# Plot 2: Land Mask
plt.subplot(1, 5, 2)
plt.imshow(np.transpose(land), cmap='cividis', extent=extent)
plt.colorbar(label='Land')
plt.title("Land")

# Plot 3: Temperature Gradient
plt.subplot(1, 5, 3)
plt.imshow(np.transpose(T), cmap='coolwarm', extent=extent)
plt.colorbar(label='Temperature')
plt.title("Temperature")

# Plot 4: Animal Distribution on Environmental Map
plt.subplot(1, 5, 4)
plt.imshow(np.transpose(np.log10(biomass_density_plot)), cmap='inferno', alpha=0.5, extent =extent)
plt.scatter(longitude, latitude, c=np.log10(M), cmap='viridis', s=5, edgecolor="k", alpha=0.7)
plt.colorbar(label="Log10(Adult Body Mass)")
plt.title("Animal Distribution")

# Plot 5: Animal Distribution on Environmental Map
plt.subplot(1, 5, 5)
plt.imshow(np.transpose(resources_plot), cmap='inferno', alpha=0.5, extent=extent)
plt.scatter(longitude, latitude, c=np.log10(R_o), cmap='viridis', s=20, edgecolor="k", alpha=0.7)
plt.colorbar(label="Log10(BasalMetRate_mLO2hr)")
plt.title("Animal Distribution")


# Show plots
plt.tight_layout()
plt.show()   

# biomass_sum_lat = np.nansum(biomass_density, axis=0)
# # Create the plot
# plt.figure(figsize=(10, 6))

# # Plot actual biomass (M) against latitude
# plt.plot(latitude, M, label='Actual Biomass', color='blue', linestyle='-', marker='o')

# # Plot modeled biomass (biomass_density) against latitude
# plt.plot(lat, biomass_sum_lat, label='Modeled Biomass', color='red', linestyle='-', marker='x')

# # Add labels and title
# plt.xlabel('Latitude')
# plt.ylabel('Biomass (g C / m^{2})')
# plt.title('Actual vs Modeled Biomass')

# # Show the legend
# plt.legend()

# # Display the plot
# plt.show()

# biomass_sum_lon = np.nansum(biomass_density, axis=1)
# plt.figure(figsize=(10, 6))

# # Plot actual biomass (M) against latitude
# plt.plot(longitude, M, label='Actual Biomass', color='blue', linestyle='-', marker='o')

# # Plot modeled biomass (biomass_density) against latitude
# plt.plot(lon, biomass_sum_lon, label='Modeled Biomass', color='red', linestyle='-', marker='x')

# # Add labels and title
# plt.xlabel('Longitude')
# plt.ylabel('Biomass kg')
# plt.title('Actual vs Modeled Biomass')

# # Show the legend
# plt.legend()

# # Display the plot
# plt.show()


