# Licență — Segmentare automată a vehiculelor (PromptCNN + SAM3)

Sistem de segmentare a vehiculelor în imagini aeriene, care antrenează o rețea
PromptCNN să genereze prompturi punctuale pentru modelul fundațional SAM3, prin
învățare cu întărire (REINFORCE / B-SCST).

## Structura proiectului

```
best_model.pth              modelul PromptCNN antrenat (vezi mai jos — nu e în repo)
SAM_Clean.ipynb             antrenarea PromptCNN (REINFORCE + SAM3)
DataSet_Making_Cars*.{ipynb,py}   construirea setului de date din UAVid
interfata/
  backend/                  API FastAPI (app.py, inference.py, model_utils.py)
  frontend/                 interfața web (HTML/CSS/JS)
PozeInterfata/              imagini + adnotări pentru interfață
experiment/                 experimentele comparative (pipeline vs box direct)
```

## Configurare

1. Instalează dependențele:
   ```
   pip install -r interfata/backend/requirements.txt
   ```
2. Creează `interfata/backend/.env` pornind de la `.env.example` și pune
   token-ul tău Hugging Face (necesar pentru descărcarea SAM3):
   ```
   HF_TOKEN=...
   ```
3. Descarcă modelul antrenat `best_model.pth` și pune-l în rădăcina proiectului.
   (Modelul nu este inclus în repo deoarece depășește limita GitHub de 100 MB.
   Link de download: [de completat].)

## Rulare interfață

```
cd interfata/backend
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```
Interfața este accesibilă la http://localhost:8000

## Experimente comparative

```
cd experiment
python run_experiment.py --lo 0.30 --hi 0.45
```
Rezultatele (matrice de confuzie, tabele, vizualizări) se salvează în
`experiment/exp_<lo>_<hi>/`.