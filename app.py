from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Cargar modelo entrenado
MODEL_PATH = os.path.join("model", "heart_disease_model.pkl")
model = joblib.load(MODEL_PATH)

# Guardamos los nombres de columnas EXACTOS que el modelo espera
FEATURE_NAMES = list(getattr(model, "feature_names_in_", []))
N_FEATS = len(FEATURE_NAMES)

def parse_float(value: str) -> float:
    """Convierte texto a float aceptando coma o punto."""
    return float(value.replace(",", ".").strip())

@app.route("/")
def home():
    return render_template(
        "index.html",
        n_features=N_FEATS,
        prediction_text=None,
        error_text=None,
        last_values={}
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        form = request.form

        # 1) Leer campos "bonitos" del formulario
        edad     = parse_float(form.get("edad", ""))
        trestbps = parse_float(form.get("trestbps", ""))
        chol     = parse_float(form.get("chol", ""))
        fbs      = form.get("fbs")          # "0" o "1"
        thalach  = parse_float(form.get("thalach", ""))
        oldpeak  = parse_float(form.get("oldpeak", ""))

        sexo     = form.get("sexo")         # "male" o "female"
        cp       = form.get("cp")           # ej. "typical angina"
        restecg  = form.get("restecg")      # "0","1","2"
        exang    = form.get("exang")        # "0","1"
        slope    = form.get("slope")        # "1","2","3"
        ca       = form.get("ca")           # "0.0","1.0","2.0","3.0"
        thal     = form.get("thal")         # "3.0","6.0","7.0"

        # Validación rápida
        if None in [sexo, cp, restecg, exang, slope, ca, thal, fbs]:
            raise ValueError("Faltan campos por llenar")

        # 2) Construir un diccionario de características internas
        #    (todas las columnas que el modelo espera)
        feats = {}

        for fname in FEATURE_NAMES:
            # Continuas directas
            if fname in ("age", "edad"):
                feats[fname] = edad
            elif fname in ("trestbps", "resting_blood_pressure"):
                feats[fname] = trestbps
            elif fname in ("chol", "cholesterol"):
                feats[fname] = chol
            elif fname in ("thalach", "max_heart_rate"):
                feats[fname] = thalach
            elif fname == "oldpeak":
                feats[fname] = oldpeak
            elif fname in ("fbs", "fasting_blood_sugar"):
                feats[fname] = float(fbs)

            # One-hot de sexo: sex_female, sex_male
            elif fname.startswith("sex_"):
                suffix = fname.split("_", 1)[1]  # "male" / "female"
                feats[fname] = 1.0 if sexo == suffix else 0.0

            # One-hot de tipo de dolor: cp_xxx
            elif fname.startswith("cp_"):
                # valores del select cp coinciden con el sufijo ("typical angina", etc.)
                feats[fname] = 1.0 if fname == f"cp_{cp}" else 0.0

            # One-hot de ECG en reposo
            elif fname.startswith("restecg_"):
                feats[fname] = 1.0 if fname == f"restecg_{restecg}" else 0.0

            # One-hot de angina por ejercicio
            elif fname.startswith("exang_"):
                feats[fname] = 1.0 if fname == f"exang_{exang}" else 0.0

            # One-hot de pendiente ST
            elif fname.startswith("slope_"):
                feats[fname] = 1.0 if fname == f"slope_{slope}" else 0.0

            # One-hot de número de vasos
            elif fname.startswith("ca_"):
                feats[fname] = 1.0 if fname == f"ca_{ca}" else 0.0

            # One-hot de Thal
            elif fname.startswith("thal_"):
                feats[fname] = 1.0 if fname == f"thal_{thal}" else 0.0

            # Cualquier columna rara la dejamos en 0
            else:
                feats[fname] = 0.0

        # 3) Convertir al vector en el orden correcto
        x_vector = np.array([[feats[f] for f in FEATURE_NAMES]])

        # 4) Predicción
        y = int(model.predict(x_vector)[0])
        proba = None
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(x_vector)[0][y])

        msg = "Tiene riesgo de enfermedad cardiaca" if y == 1 else "No tiene riesgo de enfermedad cardiaca"
        if proba is not None:
            msg += f" (confianza: {proba:.1%})"

        return render_template(
            "index.html",
            n_features=N_FEATS,
            prediction_text=msg,
            error_text=None,
            last_values=form
        )

    except ValueError as e:
        return render_template(
            "index.html",
            n_features=N_FEATS,
            prediction_text=None,
            error_text="Ingresa números válidos en todos los campos numéricos.",
            last_values=request.form
        )
    except Exception as e:
        return render_template(
            "index.html",
            n_features=N_FEATS,
            prediction_text=None,
            error_text=f"Ocurrió un error: {e}",
            last_values=request.form
        )

if __name__ == "__main__":
    app.run(debug=True)