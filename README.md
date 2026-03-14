# Physics-Informed Neural Networks (PINNs) for Climate-Driven Evolutionary Dynamics in Epidemics

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

## Overview
This repository contains a deep learning framework designed to solve the "inverse problem" in disease ecology. Traditional epidemiological forecasting models treat biological transmission rates as static parameters. However, in reality, disease vectors (such as the Dengue mosquito) undergo "phenological drift" and adapt to changing environmental pressures like extreme temperature and unpredictable rainfall. 

This project utilizes a **Physics-Informed Neural Network (PINN)** built in TensorFlow to model these eco-evolutionary dynamics. By embedding a standard SEIR compartmental model directly into the neural network's loss function, the model mathematically uncovers the hidden, time-dependent transmission rate ($\beta(t)$) of Dengue fever in Colombo, Sri Lanka, driven by historical climate anomalies.

## The Mathematical Framework (SEIR Model)
Unlike standard black-box machine learning models, this PINN is constrained by the physical and biological laws of disease transmission. The hidden layers are penalized if their outputs violate the following system of Ordinary Differential Equations (ODEs):

$$\frac{dS}{dt} = -\frac{\beta(t) S I}{N}$$
$$\frac{dE}{dt} = \frac{\beta(t) S I}{N} - \sigma E$$
$$\frac{dI}{dt} = \sigma E - \gamma I$$
$$\frac{dR}{dt} = \gamma I$$

Instead of assuming $\beta$ (the transmission rate) is constant, the neural network learns $\beta(t)$ dynamically from the noisy input data (Temperature, Rainfall, Time), revealing the vector's ecological adaptation over time.

## Data Sources
The model is trained on a strictly aligned, Min-Max scaled dataset bridging meteorological and epidemiological domains:
* **Epidemiological Data:** Monthly Dengue fever case reports for the Colombo district (2010–2020), sourced from the Epidemiology Unit, Ministry of Health, Sri Lanka.
* **Meteorological Data:** Historical weather variables (Temperature 2m Mean, Precipitation Sum) corresponding to the same geographic region and timeframe.

## Repository Structure
```text
├── data/
│   ├── raw_dengue_data.csv            # Raw historical case counts
│   ├── raw_weather_data.csv           # Raw meteorological data
│   └── PINN_Dengue_Colombo_Dataset.csv # Preprocessed, aligned, and scaled dataset
├── notebooks/
│   └── 01_data_preprocessing.ipynb    # Time-series alignment and normalization pipeline
├── src/
│   ├── model.py                       # PINN architecture using tf.keras
│   ├── train.py                       # Custom @tf.function training loop with GradientTape
│   └── utils.py                       # Loss function physics calculations
├── results/
│   └── beta_evolution_plot.png        # Visualizations of the discovered beta(t) parameter
└── README.md
