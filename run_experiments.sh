#!/bin/bash

# Abilita l'uscita in caso di errore
set -e

# Carica le variabili d'ambiente (il token) dal file .env
if [ -f .env ]; then
    export $(cat .env | xargs)
else
    echo "ERRORE: File .env non trovato. Crea un file .env con il tuo HF_TOKEN."
    exit 1
fi
echo "========================================"
echo "🚀 INIZIO ESPERIMENTO MEDCLIP (ResNet50)"
echo "========================================"
python src/train.py --hf_token "$HF_TOKEN" --output_dir "runs/resnet" --loss_type medclip --vision_model_type cnn --vision_model_name resnet50 --gradient_checkpointing --vision_layers_unfrozen -1 --text_layers_unfrozen -1

echo "========================================"
echo "🚀 INIZIO ESPERIMENTO GRAM-MED (ResNet50)"
echo "========================================"
python src/train.py --hf_token "$HF_TOKEN" --output_dir "runs/resnet" --loss_type gram_med --vision_model_type cnn --vision_model_name resnet50 --gradient_checkpointing --vision_layers_unfrozen -1 --text_layers_unfrozen -1

echo "========================================"
echo "📊 GENERAZIONE DEI GRAFICI"
echo "========================================"
python plot_results.py --baseline_metrics runs/resnet/cnn/medclip/metrics.json --proposed_metrics runs/resnet/cnn/gram_med/metrics.json

echo "✅ TUTTO COMPLETATO! I grafici sono nella cartella PLOT del tuo progetto su Windows."
