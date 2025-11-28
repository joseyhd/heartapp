from flask import Flask, render_template, request
import joblib
import numpy as np
import os
import pandas as pd  # IMPORTANTE: para crear DataFrame

app = Flask(__name__)

# Cargar modelo (Pipeline completo)
MODEL_PATH = os.path.join("model", "heart_disease_rfmodel.pkl")
model = joblib.load(MODEL_PATH)

# Obtener nombres esperados
FEATURE_NAMES = list(getattr(model, "feature_names_in_", model.feature_names_in_))
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

        # 1) Leer campos numéricos
        edad     = parse_float(form.get("edad", ""))
        trestbps = parse_float(form.get("trestbps", ""))
        chol     = parse_float(form.get("chol", ""))
        thalach  = parse_float(form.get("thalach", ""))
        oldpeak  = parse_float(form.get("oldpeak", ""))

        # 2) Convertir categóricos
        sexo     = form.get("sexo")
        cp       = form.get("cp")
        fbs      = form.get("fbs")
        restecg  = form.get("restecg")
        exang    = form.get("exang")
        slope    = form.get("slope")
        ca       = form.get("ca")
        thal     = form.get("thal")

        # Resumen para mostrar en pantalla
        resumen_datos = {
            "Edad": edad,
            "Presión arterial (mmHg)": trestbps,
            "Colesterol (mg/dl)": chol,
            "Frecuencia cardíaca máx.": thalach,
            "Depresión ST (oldpeak)": oldpeak,
            "Sexo": sexo,
            "Tipo de dolor en pecho": cp,
            "Glucosa ayunas >120": fbs,
            "ECG reposo": restecg,
            "Angina por ejercicio": exang,
            "Pendiente ST": slope,
            "N° vasos coloreados": ca,
            "Thal": thal
        }

        # Validación
        if None in [sexo, cp, restecg, exang, slope, ca, thal, fbs]:
            raise ValueError("Faltan campos por llenar")

        # 3) Crear DataFrame
        input_df = pd.DataFrame([{
            "age": edad, "sex": sexo, "cp": cp, "trestbps": trestbps, "chol": chol,
            "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
            "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
        }])

        # 4) Predicción
        y_pred = int(model.predict(input_df)[0])
        proba = model.predict_proba(input_df)[0][y_pred] if hasattr(model, "predict_proba") else None

        msg = ("Tiene riesgo de enfermedad cardiaca, acuda a un médico a la brevedad"
               if y_pred == 0 else
               "No hay evidencia suficiente para determinar riesgo cardíaco significativo, consulte a su médico")

        if proba is not None:
            msg += f" (confianza: {proba:.1%})"

        return render_template("index.html",
                               prediction_text=msg,
                               resumen=resumen_datos,
                               error_text=None,
                               last_values=form)

    except ValueError:
        return render_template("index.html",
                               prediction_text=None,
                               resumen=None,
                               error_text="Ingresa valores numéricos válidos.",
                               last_values=request.form)

    except Exception as e:
        return render_template("index.html",
                               prediction_text=None,
                               resumen=None,
                               error_text=f"Error interno: {e}",
                               last_values=request.form)
if __name__ == "__main__":
    app.run(debug=True)
