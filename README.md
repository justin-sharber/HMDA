# HMDA
Analysis of Home Mortgage Data

This is a data analysis project of home mortgage data.  This repository consists of the data files in CSV format, the Python files for computation, and [a report on the findings](https://github.com/JShibby/HMDA/blob/master/Executive%20Report%20-%20HMDA%20Data.pdf) as a PDF.

The project has two parts, and the Python files break down into two sets.  

## 1. Predictability
The first part examines predictability.  The associated Python files are:
* data.py - Loads, cleans, and encodes the data.  Contains a parameter for feature selection based on independent predictive accuracy.
* explore.py - Prepares visualizations of the data, especially for the report.
* model.py - Trains and tests various models on the data.

## 2. Discrimination
The second part examines the possibility of discrimination in home mortgages.  The associated Python file is:
* discrimination.py - Processes data and models to look for discrimination.  Contains a parameter for whether to include "unfair" (discriminatory) features.

## Note!
The original dataset is 500,000 data.  Due to size constraints, I have uploaded a subset of the data.  Computations should be close to those shown in the report, but expect differences.
