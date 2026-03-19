# import streamlit as st
# import pandas as pd
# import numpy as np
# import sklearn
# import matplotlib.pyplot as plt
# import seaborn as sns

# print(f"Pandas version: {pd.__version__}")
# print(f"NumPy version: {np.__version__}")
# print(f"Scikit-learn version: {sklearn.__version__}")
# print(f"Seaborn version: {sns.__version__}")
# print("All libraries imported successfully!")


import pandas as pd
import sklearn
import imblearn
import optuna
import shap
import streamlit as st
import matplotlib
import mlflow
import numpy as np
from pkg_resources import parse_version

requirements = {
    "pandas": "1.5.0",
    "scikit-learn": "1.2.0",
    "imbalanced-learn": "0.10.0",
    "optuna": "3.0.0",
    "shap": "0.42.0",
    "streamlit": "1.28.0",
    "matplotlib": "3.6.0",
    "mlflow": "2.5.0",
    "numpy": "1.23.0"
}

installed = {
    "pandas": pd.__version__,
    "scikit-learn": sklearn.__version__,
    "imbalanced-learn": imblearn.__version__,
    "optuna": optuna.__version__,
    "shap": shap.__version__,
    "streamlit": st.__version__,
    "matplotlib": matplotlib.__version__,
    "mlflow": mlflow.__version__,
    "numpy": np.__version__
}

print(f"{'Package':<20} | {'Installed':<10} | {'Required':<10} | {'Status'}")
print("-" * 60)

for pkg, req_ver in requirements.items():
    inst_ver = installed[pkg]
    status = "✅ OK" if parse_version(inst_ver) >= parse_version(req_ver) else "❌ UPDATE NEEDED"
    print(f"{pkg:<20} | {inst_ver:<10} | >={req_ver:<8} | {status}")


#✅ cheked all requirements on 19-03-26 12.21 pm