# Production-Based-Energy-Prediction-for-Sustainable-Manufacturing
#This was done in collaboration with Unilever PLC Sapugaskanda Sri Lanka

![image](https://github.com/Pavanirt/Production-Based-Energy-Prediction-for-Sustainable-Manufacturing/assets/160448544/51b1f2b7-8316-4df7-b7c1-53a92c37255e)

# A Random forest model was build integrating with auser input.

# %matplotlib inline
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Dictionary mapping numbers to options
options_dict = {
    1: "Viva 400g Pouch",
    2: "Viva 400g BIB",
    3: "Viva 600g Pouch",
    4: "Viva 800g Pouch",
    5: "Viva 175g Pouch",
    6: "Viva 175g BIB",
    7: "Viva 250g BIB",
    8: "Horlicks 350g",
    9: "Horlicks 200g BIB",
    10: "Horlicks 175g BIB",
    11: "Horlicks 400g BIB",
    12: "Horlicks Chocolate BIB 400g",
    13: "Junior Horlicks Vanila BIB 400g",
    14: "Viva sachet 26g",
    15: "Horlicks sachet 25g",
    16: "Viva Bottle 400g",
    17: "Horlicks Bottle 350g",
    18: "Horlicks Bottle 400g"
}

# Initialize totals dictionary to accumulate results
totals_dict = {
    'Total Weight': 0,
    'Total HVAC Energy': 0,
    'Total Lighting Energy': 0,
    'Total Air Curtain Energy': 0,
    'Total Humidity': 0,
    'Total Mixing Energy': 0,
    'Total Packing Energy': 0,
    'Total Predicted Energy': 0
}

results_dict = {}

# Load the dataset
df = pd.read_csv('/content/import data.csv')

# Selecting the features and target
X = df[['Total(Tonnage)', 'Total(Energy-Mixing)', 'Total(Packing)', 'Total(HVAC)', 'Total(Lightning)', 'Total(Air curtain)', 'Total(Dihimudity)']]
y = df['Actual Energy']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model initialization
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Model training with the training set
rf_model.fit(X_train, y_train)

while True:
    # Display options to the user
    for number, option in options_dict.items():
        print(f"{number}: {option}")

    # Input from the user
    user_input_number = input("Enter the number of your choice (or 'exit' to quit): ")

    # Check if the user wants to exit
    if user_input_number.lower() == 'exit':
        print("Exiting the program. Goodbye!")
        break

    # Check if the input is a valid number
    if not user_input_number.isdigit():
        print("Invalid input. Please enter a valid number or 'exit'.")
        continue

    user_input_number = int(user_input_number)

    # Check if the input number corresponds to a valid option
    if user_input_number in options_dict:
        selected_option = options_dict[user_input_number]
        print("Valid choice! You chose:", selected_option)

        # Input from the user: Number of cases
        num_cases = int(input("Enter the number of cases you want to process: "))
        weights = [0.0004, 0.0004, 0.0006, 0.0008, 0.000175, 0.000175, 0.00025, 0.00035, 0.0002, 0.000175, 0.0004, 0.0004,
                   0.0004, 0.000026, 0.000025, 0.0004, 0.00035, 0.0004]

        weight_index = user_input_number - 1  # Adjust index to match the weights list
        case_weight = weights[weight_index]
        total_weight = num_cases * case_weight
        print(f"Total Weight for {num_cases} cases of {selected_option}: {total_weight} tons")

        # HVAC energy for each option in ton
        hvac_energy = [
            50.8, 50.8, 50.8, 50.8, 50.8, 50.8, 50.8, 70.8, 70.8, 70.8, 70.8, 70.8, 70.8, 50.8, 70.8, 50.8, 70.8, 70.8
        ]
        hvac_energy_value = hvac_energy[weight_index]
        total_hvac_energy = total_weight * hvac_energy_value
        print(f"Total HVAC Energy for {num_cases} cases of {selected_option}: {total_hvac_energy} KW")

        # Lighting Energy values in ton
        lighting_energy_values = [
            3.466, 3.466, 3.466, 3.466, 3.466, 3.466, 3.466, 5.644, 5.644, 5.644, 5.644, 5.644, 5.644, 3.466, 5.644, 3.466,
            5.644, 5.644
        ]
        lighting_energy = lighting_energy_values[weight_index]
        total_lighting_energy = total_weight * lighting_energy
        print(f"Total Lighting Energy for {num_cases} cases of {selected_option}: {total_lighting_energy} KW")

        # Air Curtain Energy values in ton
        air_curtain_energy_values = [
            1.47, 1.47, 1.47, 1.47, 1.47, 1.47, 1.47, 2.69, 2.69, 2.69, 2.69, 2.69, 2.69, 1.47, 2.69, 1.47, 2.69, 2.69
        ]
        air_curtain_energy = air_curtain_energy_values[weight_index]
        total_air_curtain_energy = total_weight * air_curtain_energy
        print(f"Total Air Curtain Energy for {num_cases} cases of {selected_option}: {total_air_curtain_energy} KW")

        # Humidity values
        humidity_values = [
            27, 27, 27, 27, 27, 27, 27, 30, 30, 30, 30, 30, 30, 27, 30, 27, 30, 30
        ]
        humidity = humidity_values[weight_index]
        total_humidity = total_weight * humidity
        print(f"Total Humidity Energy for {num_cases} cases of {selected_option}: {total_humidity} tons")

        # Mixing Energy values in ton
        mixing_energy_values = [
            12.63, 12.63, 12.63, 12.63, 12.63, 12.63, 12.63, 46.505, 46.505, 46.505, 46.505, 46.505, 46.505, 12.63, 46.505,
            12.63, 46.505, 46.505
        ]
        mixing = mixing_energy_values[weight_index]
        total_mixing_energy = total_weight * mixing
        print(f"Total Mixing Energy for {num_cases} cases of {selected_option}: {total_mixing_energy} tons")

        # Packing Energy values in ton
        packing_energy_values = [
            15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 8, 8, 11, 11, 11
        ]
        packing_energy = packing_energy_values[weight_index]
        total_packing_energy = total_weight * packing_energy
        print(f"Total Packing Energy for {num_cases} cases of {selected_option}: {total_packing_energy} tons")

        # Predict energy consumption using the machine learning model
        input_features = np.array([[total_weight, total_mixing_energy, total_packing_energy,
                                    total_hvac_energy, total_lighting_energy,
                                    total_air_curtain_energy, total_humidity]])

        predicted_energy = rf_model.predict(input_features)
        print(f"Predicted Total Energy Consumption for {num_cases} cases of {selected_option}: {predicted_energy[0]} KW")

        # Update totals dictionary
        totals_dict['Total Weight'] += total_weight
        totals_dict['Total HVAC Energy'] += total_hvac_energy
        totals_dict['Total Lighting Energy'] += total_lighting_energy
        totals_dict['Total Air Curtain Energy'] += total_air_curtain_energy
        totals_dict['Total Humidity'] += total_humidity
        totals_dict['Total Mixing Energy'] += total_mixing_energy
        totals_dict['Total Packing Energy'] += total_packing_energy
        totals_dict['Total Predicted Energy'] += predicted_energy[0]

        # Update results dictionary
        results_dict[selected_option] = {
            'Total Weight': total_weight,
            'Total HVAC Energy': total_hvac_energy,
            'Total Lighting Energy': total_lighting_energy,
            'Total Air Curtain Energy': total_air_curtain_energy,
            'Total Humidity': total_humidity,
            'Total Mixing Energy': total_mixing_energy,
            'Total Packing Energy': total_packing_energy,
            'Predicted Energy': predicted_energy[0]
        }

    else:
        print("Invalid choice. Please enter a valid number.")

# Display final summary of totals
print("\nSummary of Totals:")
for key, value in totals_dict.items():
    print(f"{key}: {value}")

# Print the results dictionary
print("\nResults:")
for option, values in results_dict.items():
    print(f"\nOption: {option}")
    for key, value in values.items():
        print(f"{key}: {value}")

#Visualization of Insights
![image](https://github.com/Pavanirt/Production-Based-Energy-Prediction-for-Sustainable-Manufacturing/assets/160448544/3d90d8e8-7139-492c-a920-d4a7fce38d1a)
