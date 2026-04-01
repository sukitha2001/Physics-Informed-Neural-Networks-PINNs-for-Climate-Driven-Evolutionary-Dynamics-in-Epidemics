import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def prepare_pinn_dataset():
    print("Loading raw datasets...")
    # Load the datasets
    df_dengue = pd.read_excel("../Data/Dengue_Data (2010-2020).xlsx")
    df_weather = pd.read_csv("../Data/SriLanka_Weather_Dataset.csv")

    # ==========================================
    #Time Alignment
    # ==========================================
    print("Aligning timelines...")
    # Format Dengue dates and rename the target column
    df_dengue['Date'] = pd.to_datetime(df_dengue['Date'])
    df_dengue.rename(columns={'Value': 'Dengue_Cases'}, inplace=True)

    # Format Weather dates and round down to the start of the month
    df_weather['time'] = pd.to_datetime(df_weather['time'])
    df_weather['Date'] = df_weather['time'].dt.to_period('M').dt.to_timestamp()

    # Aggregate daily weather into monthly averages/sums
    weather_monthly = df_weather.groupby(['Date', 'city']).agg(
        Temperature=('temperature_2m_mean', 'mean'),
        Rainfall=('precipitation_sum', 'sum')
    ).reset_index()

    # ==========================================
    #Geographic Isolation & Merging
    # ==========================================
    print("Merging data for Colombo...")
    # Standardize city names to prevent mismatch errors
    df_dengue['City'] = df_dengue['City'].str.strip().str.title()
    weather_monthly['city'] = weather_monthly['city'].str.strip().str.title()

    # Merge the two datasets mathematically on Date and City
    merged_df = pd.merge(
        df_dengue, 
        weather_monthly, 
        left_on=['Date', 'City'], 
        right_on=['Date', 'city'], 
        how='inner'
    )
    
    # Isolate just Colombo for the PINN continuous environment
    colombo_df = merged_df[merged_df['City'] == 'Colombo'].copy()
    colombo_df.sort_values('Date', inplace=True)

    # ==========================================
    # Creating the Physics Time Vector (t)
    # ==========================================
    # Generate the continuous numeric tensor for the ODE calculations
    colombo_df['Time_Step'] = np.arange(len(colombo_df), dtype=float)

    # ==========================================
    #Min-Max Scaling
    # ==========================================
    print("Applying Min-Max scaling for deep learning gradients...")
    scaler = MinMaxScaler()
    
    # Compress all inputs to sit strictly between 0.0 and 1.0
    cols_to_scale = ['Temperature', 'Rainfall', 'Dengue_Cases']
    colombo_df[['Temp_Scaled', 'Rain_Scaled', 'Cases_Scaled']] = scaler.fit_transform(colombo_df[cols_to_scale])
    
    # Reorder columns for a clean, logical export
    final_df = colombo_df[[
        'Date', 'City', 'Time_Step', 
        'Temperature', 'Rainfall', 'Dengue_Cases', 
        'Temp_Scaled', 'Rain_Scaled', 'Cases_Scaled'
    ]]
    
    # Export the dataset
    output_filename = "../Data/Cleaned_Dataset.csv"
    final_df.to_csv(output_filename, index=False)
    
    print(f"\nSuccess! dataset saved as: {output_filename}")
    print(final_df.head())

# Execute the function
prepare_pinn_dataset()