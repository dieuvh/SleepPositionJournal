# SleepPositionJournal

Static sensor time-series classification based on `Jounar_2023` from phamdinhdat-ai/Time_series_project.

Data intentionally **not uploaded**: the source .npy files have participant-identifying filenames. Supply your local dataset under `data/new_data_static/`.

## Standard person split
```bash
pip install -r requirements.txt
python main.py --model_type adaptive_lstm --scenario person_divide --data_type static --num_classes 12 --epochs 100 --sequence_length 100 --overlap 0.4
```

Uses `data/new_data_static/trainset` and `testset`.

## Subject K-fold
```bash
python main.py --model_type adaptive_lstm --scenario person_divide --data_type static --num_classes 12 --epochs 100 --sequence_length 100 --overlap 0.4 --k_fold 5
```

Uses `data/new_data_static/all_person`; each fold trains a fresh model on disjoint subjects, fits scaler only on train subjects, and forms windows **within contiguous same-label runs**. Saves per-fold reports and mean/std to `results/`.

This is the static person_divide training path; legacy sample_divide, Comet and Lipschitz logging are not included. Full TensorFlow training has **not** been executed or benchmarked on the real dataset.
