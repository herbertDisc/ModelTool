# ModelTool

Various tools for modeling.

## Tools explanation

- Model_report.py : model_reporter() to generate a performance report for the model, which contains model overview, bivar, gains table, means table.
- model_card_html.py : render_model_card() turns a model_reporter workbook (model_N.xlsx) into a single-file interactive HTML model card; model_building writes this card in place of the workbook. CLI: `python -m modeltool.model_card_html report/model_*.xlsx`.
- model_tracking.py : production monitoring tables benchmarked on each model's first month.
    - model_tracker() one model: daily score count/avg/PSI and attribute PSI (frontend), weekly AUC/KS, score and attribute PSI, IV shift and gains table (backend).
    - track_models() several models in one go; saves the stacked tables and the HTML report to one folder.
- tracking_report.py : build_report() self-contained HTML tracking report from those tables (overview + one tab per model, attribute issue finder). Rebuild from a saved folder with `python -m modeltool.tracking_report <folder>`.
- model_building.py : 
    - train_single_model() train a single model
    - hyperopt_search() random search for a single round
    - iter_random_search() iteratively doing random search until the # of attributes meet the criteria.