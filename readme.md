# H2O Analytics: Marketing for Water Quality Prediction

## Project Summary
In the agriculture industry, poor water quality can significantly decrease crop yield and increase soil contamination. H2O Analytics will develop a machine learning model capable of predicting water quality as "good" or "bad." Our predictive model will enable the optimization of plant growth and adherence to environmental regulations.

Predicting water quality can greatly benefit the crop yield and the environment by ensuring only clean water is used for irrigation, thereby reducing soil and plant contamination. H2O Analytics’ model will predict water quality based on key features including levels of pH, iron, nitrate, chloride, lead, zinc, color, turbidity, fluoride, copper, odor, sulfate, conductivity, chlorine, manganese, total dissolved solids. Factors such as source water temperature, air temperature, month, day, and time of day will also be considered.

After exploring which features are the primary factors leading to contamination, various models will be created and tested to determine the optimal fit for this prediction problem. Our model will allow us to sell services to farmers, such as optimal times of day to water plants for their specific environmental factors, as well as recommendations for water filtration. We are also able to sell our model to companies selling filtration systems, so they may optimize their marketing to farmers in poor water quality areas. Finally, the government’s environmental agencies are another potential client of ours. The EPA has a responsibility to keep an eye on water quality, and knowing potential locations of contamination could push investigations into methods for cleanup.

H2O Analytics hopes to not only target farmers, water filtration companies, and governmental agencies but also to contribute to the sustainable use of water and farming techniques.

## Link to Dataset
[Water Quality Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/vanthanadevi08/water-quality-prediction)

## Source
“Water Quality for Crop Production.” University of Massachusetts Amherst, Center for Agriculture, Food, and the Environment, 19 Apr. 2019. [Link](https://ag.umass.edu/greenhouse-floriculture/greenhouse-best-management-practices-bmp-manual/water-quality-for-crop)


## Using Notebook

### Clone the repo
```bash
git clone https://github.com/sebastien-andre/Water_Quality_Project.git
cd Water_Quality_Project
```

### Set up the env
- On Mac
```bash
python3 -m venv env
source env/bin/activate
```

- On Windows
```bash
python -m venv env
.\env\Scripts\activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

