# Weather
# Weather Classification Project

## Overview

This project uses a dataset of daily weather observations across Australia over a period of 10 years. The goal of the project is to predict if it will rain tomorrow (`RainTomorrow`), based on weather features from the previous day.

The dataset contains multiple columns, including features such as temperature, humidity, wind speed, and more. The target column `RainTomorrow` is a binary value (Yes/No), indicating if it will rain at least 1mm on the next day.

## Dataset

- **Source**: [Bureau of Meteorology](http://www.bom.gov.au/climate/data)
- **Target variable**: `RainTomorrow` (Yes/No)
- **Features**: Weather observations such as temperature, wind, humidity, etc.
## Downloading the Data

The dataset is available at [Dataset](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package).
The .ipynb file is 
[WeatherClassifier.ipynb](./WeatherClassifier.ipynb)

### Key Columns:
- **Date**: The date of observation
- **Location**: The location of the weather station
- **MinTemp**: Minimum temperature
- **MaxTemp**: Maximum temperature
- **Rainfall**: Amount of rainfall in mm
- **WindSpeed9am/3pm**: Wind speed at 9 AM and 3 PM
- **Humidity9am/3pm**: Humidity at 9 AM and 3 PM
- **Pressure9am/3pm**: Pressure at 9 AM and 3 PM
- **RainToday**: Whether it rained today (Yes/No)
- **RainTomorrow**: Whether it will rain tomorrow (Yes/No) - target variable

## Goal

The objective is to predict whether it will rain tomorrow based on historical weather data using classification models. The project aims to provide accurate predictions that help answer the question: **"Should I carry an umbrella tomorrow?"**

## Models Used

Several classification algorithms were applied to the dataset, including:
- **Random Forest Classifier**
- **Logistic Regression**
- **Decision Trees**

## Best Model

- **Random Forest Classifier**
- **Accuracy**: 85.7%

