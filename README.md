# ModelTool

Various tools for modeling.

## Tools explanation

- Model_report.py : model_reporter() to generate a performance report for the model, which contains model overview, bivar, gains table, means table.
- model_building.py : 
    - train_single_model() train a single model
    - hyperopt_search() random search for a single round
    - iter_random_search() iteratively doing random search until the # of attributes meet the criteria.