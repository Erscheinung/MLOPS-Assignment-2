We will be using ZenML for the pipelines and for AutoEDA, we'll use pandas-profiling. And we will use the same dataset from Assignment 1--Iris dataset because of its simplicity: 4 features and 3 classes as well as no missing values/outliers.


Install Dependencies: `pip install -r requirements.txt`

Run `zenml init` to initialize zenML.

To run the entire pipeline, run:

`python src/preprocessing.py`

`python src/model_selection.py`

`python src/model_explanation.py`


It getting issues while running on a conda environment, run:

`python -m src.preprocessing`

`python -m src.model_selection`

`python -m src.model_explanation`

To view the zenml dashboard:

`zenml up --blocking` on a separate terminal

run `zenml connect --url <ENTER URL FROM ABOVE OUTPUT>`

Credentials:

- username: default
- password: [BLANK, press ENTER]

Screenshots:

[ZenML preprocessing script logs](Terminal-Preprocessing.png)

[ZenML Stacks](/ZenML%20dashboard.png)