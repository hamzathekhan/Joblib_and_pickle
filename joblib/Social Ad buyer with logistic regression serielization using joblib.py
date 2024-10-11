

import joblib
joblib.dump(model,"model_joblib")
mj = joblib.load("/content/model_joblib")
mj.predict([[45,80000]])
