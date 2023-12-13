# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 00:47:01 2023

@author: Sana Khan
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


def energy_consumption(df):
    """reads data on CO2 emission from world Bank file   
       Parameters:
       - df (pd.DataFrame): DataFrame containing World Bank data.
        Returns:
       - df1 (pd.DataFrame): Filtered DataFrame for energy consumption.
       - df1_transposed (pd.DataFrame): Transposed DataFrame for analysis."""

    df1 = df.loc[df["Series Name"] ==
                 "Energy use (kg of oil equivalent per capita)"]
    df1_transposed = df1.transpose()
    return df1, df1_transposed


def CO2_emission(df):
    """reads data on CO2 emission from world Bank file
    Parameters:
    - df (pd.DataFrame): DataFrame containing World Bank data.

    Returns:
    - df1_CO2 (pd.DataFrame): Filtered DataFrame for CO2 emissions.
    - df2_CO2_transposed (pd.DataFrame): Transposed DataFrame for analysis.
"""

    df1_CO2 = df.loc[df["Series Name"] ==
                     "CO2 emissions (metric tons per capita)"]
    df2_CO2_transposed = df1_CO2.transpose()
    return df1_CO2, df2_CO2_transposed


def GDP(df):
    """reads GDP data from  file and returns 2 GDP dataframes 
    Parameters:
  - df (pd.DataFrame): DataFrame containing World Bank data.

  Returns:
  - df1_GDP (pd.DataFrame): Filtered DataFrame for GDP.
  - df2_GDP_transposed (pd.DataFrame): Transposed DataFrame for analysis.
"""
    df1_GDP = df.loc[df["Series Name"] == "GDP (current US$)"]
    df2_GDP_transposed = df1_GDP.transpose()
    return df1_GDP, df2_GDP_transposed


def Electric_consumption(df):
    """ Reads data on electric power consumption from the World Bank file.

     Parameters:
     - df (pd.DataFrame): DataFrame containing World Bank data.

     Returns:
     - df1_elec (pd.DataFrame): Filtered DataFrame for electric power 
     consumption.
     - df2_elec (pd.DataFrame): Transposed DataFrame for analysis.
"""

    df1_elec = df.loc[df["Series Name"] ==
                      "Electric power consumption (kWh per capita)"]
    df2_elec = df1_elec.transpose()
    return df1_elec, df2_elec


    # checks whether the code is imported or run directly.
if __name__ == "__main__":
    # Reading World Bank data from a CSV file
    df = pd.read_csv("Worldbankdata.csv", skip_blank_lines=True,
                     index_col="Country Name",
                     usecols=[0, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                              17, 18])
    # Extracting and analyzing energy consumption data
    energy_df1, energy_df2 = energy_consumption(df)
    print("Statistical properties of energy consumption data :\n",
          energy_df1.describe())
    # Extracting and analyzing CO2 emission data
    CO2_df1, CO2df1_transposed = CO2_emission(df)
    stat_prop_CO2_data = CO2_df1.describe()
    print("CO2 emission statistical properties:\n", stat_prop_CO2_data)
    # Calculating and printing average and standard deviation for CO2 emission
    aver = CO2_df1.describe(include='all').loc['mean']
    std = CO2_df1.describe(include='all').loc['std']
    print("average=", aver)
    print("standard Deviation=", std)
    # Extracting and analyzing GDP data
    GDP_df1, GDP_df_transposed = GDP(df)
    stat_prop_GDP_data = GDP_df1.describe()
    print("GDP data Statistical properties:\n", stat_prop_GDP_data)
    # Extracting and analyzing electric power consumption data
    df1_elec, df2_elec = (Electric_consumption(df))
    stat_prop_Elec_consumption = df1_elec.describe()
    print("Electric Consumption properties\n", stat_prop_Elec_consumption)

    # years the data was observed.
    x = np.linspace(2000, 2014, 14)

    countries = ["Brazil", "Canada", "China", "Egypt", "Germany",
                 "India", "South Africa", "Russia", "United States",
                 "Japan", "S Korea"]

    """calculates the centralised and normalised excess kurtosis and skewness
    of the dataframe"""
    # Line plots showing energy use in 11 different countries from 2000 - 2014
    for country in countries:
        skew = []
        kurt = []

        y = np.array(energy_df1.loc[country, "2000 [YR2000]":"2014 [YR2014]"])
        skewness = np.sum(((y - np.mean(y)) / np.std(y)) ** 3) / len(y - 2)
        skew.append(skewness)
        kurtosis = np.sum(((y - np.mean(y)) / np.std(y))
                          ** 4) / len(y - 3) - 3.0
        kurt.append(kurtosis)
        plt.bar(x, y, label=country, linewidth=2)

    print("skewness=", skewness)
    print("kurtosis=", kurtosis)
    plt.legend(loc="upper right", frameon=True, labelspacing=0.5)
    plt.title("Energy use")
    plt.xlabel('Year')
    plt.xlim(1998, 2022)

    plt.ylabel("kg of oil equivalent per capita")
    plt.show()

    # Line plot of CO2 emissions in 11 different countries
    for country in countries:
        z = np.array(CO2_df1.loc[country, "2000 [YR2000]":"2014 [YR2014]"])
        plt.plot(x, z, label=country, linewidth=2)

    plt.legend(loc="upper right", frameon=True, labelspacing=0.5)
    plt.title("CO2 emission")
    plt.xlabel("Year")
    plt.ylabel("Metric tons per capita")
    plt.xlim(1998, 2022)
    plt.show()

    # Line plot of GDP in 11 different countries
    for country in countries:
        data = np.array(GDP_df1.loc[country, "2000 [YR2000]":"2014 [YR2014]"])
        plt.plot(x, data, label=country, linewidth=2)

    plt.legend(loc="upper right", frameon=True, labelspacing=0.5)
    plt.title("GDP")
    plt.xlabel("Year")
    plt.xlim(1998, 2022)
    plt.ylabel("Current US$")
    plt.show()

    # line plot of electric power consumption in 11 countries
    for country in countries:
        data = np.array(df1_elec.loc[country, "2000 [YR2000]":"2014 [YR2014]"])
        plt.bar(x, data, label=country, linewidth=2, alpha=0.7)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), title="Legend")
    plt.title("Electric power consumption")
    plt.xlabel("Year")
    plt.xlim(1998, 2022)
    plt.ylabel("kWh per capita")
    plt.show()

    # plot of correlation heatmap (Canada)
    canada = df.loc["Canada"].set_index('Series Name').T
    print(canada)

    correlation_canada = canada.corr()
    print(correlation_canada)

    plt.figure(figsize=(9, 6))
    sns.heatmap(correlation_canada, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Canada correlation Map")
    plt.tight_layout()
    plt.show()
    
    # plot of correlation heatmap (China)
    correlation_china = df.loc["China"].set_index('Series Name').T
    print(correlation_china)

    correlation_china = correlation_china.corr()
    print(correlation_china)

    plt.figure(figsize=(9, 6))
    sns.heatmap(correlation_china, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("China correlation Map")
    plt.tight_layout()
    plt.show()
