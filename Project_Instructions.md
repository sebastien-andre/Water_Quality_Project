# CSC 4780/6780 and DSCI 4780 - Project Description

In the course project, you are assigned to design and implement a data analytics solution. The phases of the project will be described here. This will be a team project.

## Phase 0 – Form a Group and Propose a Project

Before you start your project, you will first form project groups. See the number of members in each team below:

- (i) If you are a PhD student, you are expected to do the project alone.
- (ii) If you are a group of graduate and/or undergraduate students - a group of 3 to 4 members.

You are free to select your team members as long as you are in the same section. If you cannot find a team (or need a team member), please let me know immediately, and I will assign you to a random group (or assign a random person to your group). After forming your group, you are expected to pick a group representative and an analytics topic. The representative will send me a confirmation email (to cpandey1@gsu.edu). The title of the email will be “CSC 6780 [Your group name] membership”. Please cc all your team members. The content of the email will be your project group name, the group members’ names, and a project summary. Project summary is a short description (~250-300 words) of the project, and it should be structured as a proposal. Therefore, please include the candidate data sources you can use, mention the problem, and your proposed approach in that summary. Once you form a team, you cannot change your team; pick your team wisely. Your project topics must be approved by me. If I believe your project topic may be too simple or too complicated, you should come to my office hours (or by appointment), then explain and convince me that your project will be original and sophisticated enough (or feasible) for this course. If you skip the approval part and your project is not satisfactory in the end (meaning that it does not meet the requirements and standards), points may be deducted from your final grade. Please note that I will disregard the email if the title does not include the membership or your group members are not cc’ed.

**Deadline for Phase 0: February 23, 2024**

## Phase 1 – Business Understanding

In the first phase of the project, you are expected to decide on the business problem. As we have learned in this class, the first phase of the data analytics projects is understanding the business. The ‘business understanding’ phase involves understanding the problem, assessing the situation, determining the data analytics goals, and creating a project plan. You are expected to formulate the business problem you proposed in Phase 0. The first step towards this task is to find a suitable dataset or datasets which you can use to perform your analytics tasks and build your solution. Note here that since you do not have an actual business problem, it would be easier for you to find a dataset first and then propose a solution. There are a number of open datasets and repositories available through the following:

1. Kaggle Datasets: https://www.kaggle.com/datasets
2. UCI Machine Learning Datasets: https://archive.ics.uci.edu/ml/
3. DrivenData: https://www.drivendata.org/
4. FiveThirtyEight Datasets: https://data.fivethirtyeight.com/
5. Microsoft Research Open Data: https://msropendata.com/
6. Awesome Public Datasets: https://github.com/awesomedata/awesome-public-datasets
7. Harvard Dataverse: https://dataverse.harvard.edu/

You may search those datasets from
- https://datasetsearch.research.google.com/
- https://search.datacite.org/
- https://www.re3data.org/search
- https://guides.library.cmu.edu/machine-learning/datasets

At the end of Phase 1, you are expected to identify and finalize the following: (1) Business Problem, (2) Dataset and general characteristics of the dataset, and (3) Proposed analytics solutions. Your solution is expected to be in the framework of a supervised learning model; however, you can explore different analytics approaches, including but not limited to unsupervised learning, deep learning, frequent pattern mining, or outlier detection. The project groups are encouraged to discuss their business and data understanding steps and get feedback. We will set up extra office hours to accommodate this. Please follow iCollege announcements.

## Phase 2 – Data Understanding and Preparation

This phase of the project will be about data exploration. Knowing your data is one of the most vital components of your data analytics solution. We do not want oversimplified solutions, but we also do not want complex, unfeasible ones. Therefore, try to confinethe number of descriptive features to 20 in this project. You can have more if they are important; however, you do not need to. If your proposal includes different data modalities (e.g., time series, images, graphs, documents, etc.), you will again need to explore the characteristics of your data. In such cases, please consult with me.

In this phase, you will explore the data, understand the characteristics of your attributes, identify missing values and outliers (then handle them accordingly), normalize your data (if necessary), and apply transformations (such as binning, mapping, aggregations, flags, etc.) and [optional] feature selection. After exploration, you will create your base table (the dataset that you will use for the learning models). This is of course if you are using tabular data. If you want to use other modalities, please make sure you follow the necessary preprocessing steps.

At the end of Phase 2, you are expected to generate a data quality report and a data quality plan. Provide explanations for how you perform the data pre-processing steps (handling outliers and missing values, selecting features, normalizations, transformations) and create your machine learning-ready dataset (analytics base table) for the next phase. You will then submit a progress report outlining the work you performed in the first two phases (a template will be provided). You will also submit your dataset before and after transformations.

## Phase 3 – Model Selection and Evaluation

After creating your analytics base table in Phase 2, the next step is training your models. To be able to evaluate your models, you need training and testing sets. Additionally, you need a meaningful metric to evaluate your results. If you have a large-scale dataset or if you have a minority class, you can use undersampling or oversampling techniques.

In this phase, you will test at least three different categories of learning models (if supervised learning, these are Information-based Learning, Similarity-based Learning, Probability-based Learning, and Error-based Learning). After selecting your learning algorithms, train your models, and test their performance using an appropriate metric (this should be based on your business task). You are not limited to using the models we have learned in the class, and you can choose to use other models which we have not covered. The same applies to the evaluation metrics.

You are expected to report the performance of your models in training and testing phases, along with the confusion matrices, and pick one as a deployment recommendation. This recommendation should not be arbitrary, and you should be able to communicate why you picked a certain model (model performance, interpretability, robustness, efficiency, ease of deployment, etc.).

## Phase 4 – Communicate Your Findings and Recommend an Action

In this phase, you will interpret the results from the model you recommended for deployment. You are expected to show the relationships between your important descriptive features and your target variable. Report the correlations that can be used as flags based on your model. Then, finally, recommend an action and discuss how this can further be integrated.

## Deliverables

Project will be worth 20 points in total (corresponding to 20% of your final grade). An additional 5 points will be available as a bonus (5% of your final grade).

### Project Progress Report & Original and Pre-processed Dataset [2+2 points]: 
You will submit your project progress report along with the original dataset that you have collected for your business objective. Additionally, you will submit your preprocessed and cleaned analytics base table or dataset. Please compress them with zip.
- The progress report must be in PDF format and named [GROUP\_NAME]_final_report.pdf.
- The raw dataset should be named [LAST_NAME]_original.zip
- The pre-processed data should be named [LAST_NAME]_pp.zip
In case these files are large, please share these files online (either through your own OneDrive/Google Drive or through public sharing services such as Zenodo or figshare). Your project will not be considered for grading without a dataset.
**Deadline: March 22, 2024**

### Project Implementation [6 points]: 
You will submit two Jupyter Notebooks, one for data exploration and preprocessing (Phase 2) and the second for modeling and evaluation (Phase 3). Make sure you cover all the requirements. If a requirement does not apply (e.g., no missing values), explicitly specify that it does not apply. Please add necessary comments and explanations with markdown cells. Make sure to check if your code runs. To do that, you can restart your kernel and run all the cells. Since we have many projects to grade, we will not attempt to interpret your intent or debug them and make it work. You will upload these files to iCollege together with your final report.
**Deadline: April 19, 2024**

### Final Report and Presentation [6+4 points]: 
You will submit a final project report which includes technical aspects of your project. You will in detail explain

