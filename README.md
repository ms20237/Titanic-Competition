# Titanic Competition

Predicts passenger survival on the Titanic (Kaggle's classic binary
classification competition) using two notebooks: one for exploration,
one for modeling.

## Folder structure

Put both notebooks in a folder like this:

```
titanic-project/
├── dataset/
│   ├── train.csv
│   └── test.csv
├── 01_EDA.ipynb
├── train_model.ipynb
├── eda_output/        ← created automatically by 01_EDA.ipynb
└── outputs/            ← created automatically by train_model.ipynb
```

## 1. Get the data

Download `train.csv` and `test.csv` from the
[Titanic competition page](https://www.kaggle.com/competitions/titanic/data)
into `dataset/`. Or with the Kaggle CLI:

```bash
pip install kaggle
kaggle competitions download titanic -p dataset --unzip
```

(Requires a Kaggle account + API token in `~/.kaggle/kaggle.json`.)

## 2. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

## 3. Run

Open both notebooks with Jupyter/VS Code and run top to bottom:

```bash
jupyter notebook
```

**`01_EDA.ipynb`** — exploration only, no modeling. Produces:
- Shape, dtypes, missing-value counts, and summary stats for train/test
- Target distribution (counts + pie chart)
- Survival rate by sex, class, embarkation port, family size, title,
  ticket prefix, and age band
- Missing-value comparison (train vs. test)
- Numeric distributions + boxplots for Age/Fare/SibSp/Parch, with a
  log-transform check on Fare's skew
- Correlation heatmap and a pairplot colored by survival
- All charts saved as `.png` files under `eda_output/`

**`train_model.ipynb`** — preprocessing, training, validation, tuning,
and submission generation. Produces:
- A reusable `add_features()` function (FamilySize, IsAlone,
  FamilySizeGroup, Title, Deck, TicketPrefix, FarePerPerson, AgePclass)
- A `run(model_key, engineered, cv)` function that builds a
  preprocessing + model pipeline, reports hold-out and cross-validation
  accuracy, and writes `outputs/submission_<model>.csv`
- A `GridSearchCV` sweep over Random Forest hyperparameters
  (`n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`)
- The tuned model refit on all training data, saved to
  `outputs/submission_rf_tuned.csv`

## Results so far

| Model | Hold-out acc | 5-fold CV acc |
|---|---|---|
| Logistic Regression + engineered features | 81.6% | 82.7% |
| Random Forest, `GridSearchCV`-tuned | — | **83.7%** |

Best parameters found: `max_depth=6`, `max_features="sqrt"`,
`min_samples_leaf=1`, `n_estimators=500`.

## Submit to Kaggle

Upload any `outputs/submission_*.csv` on the competition's
[Submit Predictions](https://www.kaggle.com/competitions/titanic/submit)
page. Required format: `PassengerId,Survived`, 418 rows.

## Notes

- Run `01_EDA.ipynb` before `train_model.ipynb` — it's not a dependency,
  but the plots make the feature choices in `add_features()` easier to
  follow.
- Both notebooks resolve paths relative to the notebook's working
  directory (`./dataset`, `./eda_output`, `./outputs`), so keep the
  folder structure above.
- To try other models, call `run(model_key="gb")` or `run(model_key="rf")`
  instead of the default `logreg`.

## License

This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/).  