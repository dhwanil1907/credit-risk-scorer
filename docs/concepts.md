# Interview answers — Credit Risk Scorer

Say these in your own words. Numbers below match the current code and the last training run.

---

## Walk me through the project (45 seconds)

I built a credit default scorer on the Home Credit dataset, about 307,000 loan applications. I joined five tables, engineered ratios and interaction features, and compared logistic regression, random forest, and XGBoost. XGBoost got about 0.78 ROC-AUC. Only about 8% of applicants default, so I did not use accuracy or a 0.5 cutoff. I tuned each model's threshold to maximize F1.

The score the user sees is a safety score from 0 to 100, higher means safer. On top of the model I added hard policy caps, because a high external credit score was marking a borrower with payments at 267% of income as low risk. Each prediction comes with the top three SHAP reasons. The dashboard calls a FastAPI service. That service is the only place the model runs. The same image runs locally in Docker, and GitHub Actions runs the tests, lint, and the image build on every push.

---

## Why not accuracy?

If I always predict "will repay," I am right about 92% of the time and I never catch a default. Accuracy rewards that. ROC-AUC asks a different question: if I pick one defaulter and one repayer at random, how often do I rank the defaulter as riskier? 0.78 means about 78% of the time. That is ranking quality, and it does not depend on a cutoff.

I still need a cutoff to approve or decline. I pick the threshold that maximizes F1 on the held-out test set, separately for each model, because precision and recall both matter and 0.5 is meaningless when defaults are rare.

---

## How did you handle the 8% default rate?

Three things, and I did not use SMOTE.

- XGBoost `scale_pos_weight` is non-defaults divided by defaults, about 11. The loss treats each default as roughly eleven repaid loans.
- Random forest and logistic regression use `class_weight="balanced"`, which does the same idea inside scikit-learn.
- The 80/20 split is stratified, so both sides keep about an 8% default rate.

SMOTE invents new default rows by interpolating real ones. Reweighting the loss leaves the data alone, which is cleaner for tree models.

---

## What data did you use, and how did you join it?

Five tables, all on `SK_ID_CURR`: the application, bureau, previous applications, installment payments, and credit card balance. Bureau and the others are many rows per person, so I aggregate to one row per applicant first, then left join. Someone with no bureau file stays in the data and gets zeros, which means "no history on file," not "dropped."

I left out POS cash balance and bureau balance. Five tables already cover application, credit file, prior loans, payment behavior, and cards.

`DAYS_EMPLOYED` uses 365243 as a placeholder for people who are not employed. I turn that into missing before filling gaps, or it would destroy the median. Other numeric gaps are filled with the median, not the mean, because income and loan size are skewed. Age and years employed are just the negative day counts divided by 365 so a person can read them.

---

## What features actually matter?

The external bureau scores, `EXT_SOURCE_1/2/3`, are the strongest raw signals. Ratios matter more than raw dollars: loan over income, yearly payment over income, and a rough loan term, credit divided by the annuity.

I added interactions after a failure. The base model scored a borrower whose payments were 2.67 times their income as low risk, because the external scores were high. So I built the average of the three external scores, multiplied it by the debt ratios, and added `DEBT_STRESS`, which is those two ratios added together. That tells the model a good credit score does not cancel an impossible payment.

---

## Why three models?

Logistic regression is the baseline. It is a linear model on the log-odds, so it cannot learn "good score times bad debt." I regularized it with `C=0.1`.

Random forest is a bag of trees on bootstrap samples. It can learn interactions, but it was the weakest of the three, about 0.74 ROC-AUC, and the file is about 300MB, so I do not serve it.

XGBoost is sequential trees. Each new tree fits the residuals of the current ensemble, shrunk by a learning rate of 0.03. I cap depth at 6, use subsample and column subsample, L2 regularization, and early stopping: it may build up to 3000 trees but stops after 100 rounds with no AUC gain on a validation slice. It won, about 0.78 ROC-AUC, and the file is 2.4MB, so that is what the API serves.

---

## What is the score the user sees?

The model outputs a default probability. Safety score is `(1 - probability) * 100`, clipped to 0–100. Higher is safer.

Then policy rules can only make it worse:

- Payments above income: cap at 30.
- Payments above half of income: cap at 55.
- Loan bigger than 10 times income: cap at 45.
- More than 60 days overdue on the bureau file: subtract 15, and never go below 0.

The UI shows the model score and the score after rules separately, with the rule text. That is the point: the model is not allowed to override a hard lending limit, and a reviewer can see which limit fired.

---

## Explain SHAP without the formula

SHAP splits one prediction into a contribution from each feature. They add up. Start from the average applicant, then each feature pushes the default probability up or down.

I use TreeExplainer because the model is a tree ensemble, so those contributions are exact and fast. I build the explainer once when the API starts, not on every request.

For the response I return the three features with the largest absolute contribution, in plain English, and whether each one helped or hurt the safety score. Positive SHAP means "pushes toward default," which hurts the safety score. The dashboard also gets the full list so it can draw the waterfall.

In lending this matters because a denial needs a reason. SHAP is a concrete reason tied to that person's inputs, not a global feature ranking.

---

## How do you keep training and the API from drifting apart?

The training script writes `outputs/feature_cols.json`, the exact column list and order. The API runs the same encode and feature functions, then aligns to that list. A missing column is filled with 0, same as "no history" at training time. If I change a feature, I retrain. I already hit this once: an older model file did not know the interaction columns, and XGBoost rejected the row until I retrained.

---

## How is it deployed?

`POST /predict` takes the applicant fields, validates them, and returns the safety score, the band, any rules that fired, and the top three reasons. Bad input is a 422 with the field name. `GET /health` is for the load balancer.

Streamlit does not load the model. It posts to the API. Docker builds one image, non-root, with only the XGBoost file. Compose runs the API on 8000 and the dashboard on 8501. GitHub Actions runs ruff, the 30 tests, and `docker build` on every push. `main` is protected: a pull request has to pass those checks.

The tests do not need the Kaggle files. They use small synthetic frames. The API tests do need `models/xgboost.pkl`, which is why that 2.4MB file is in git and the 300MB forest is not.

---

## What would you say if they push on limitations?

- 0.78 is solid for this dataset. The top of the Kaggle leaderboard is about 0.80. I did not do a big hyperparameter search or stack models.
- There are no live users. Drift monitoring would be me replaying the test set, so I did not pretend to have production traffic.
- The policy caps are business rules I chose, not a regulator's rulebook. The point of the design is that rules sit on top of the model and are visible.
- Gender is a feature because it is in the training table. I would flag that in a real lending system. Using it can be illegal. I would drop it and retrain before this was a real decision.
- I did not tune hyperparameters on the test set. The test set is only for the final comparison and the F1 threshold. The XGBoost early-stopping slice is cut from the training set.
