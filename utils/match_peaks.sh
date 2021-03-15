set -e

python utils/match_peaks.py -r results/Test/Test_ff_model_predictions.csv
python utils/match_peaks.py -r results/Test/Test_ff_model_dpo_predictions.csv
python utils/match_peaks.py -r results/Test/Test_ff_model_ta_predictions.csv
python utils/match_peaks.py -r results/Test/Test_ff_model_dpo_ta_predictions.csv
python utils/match_peaks.py -r results/Test/Test_gru_model_predictions.csv
python utils/match_peaks.py -r results/Test/Test_gru_model_dpo_predictions.csv
python utils/match_peaks.py -r results/Test/Test_gru_model_ta_predictions.csv
python utils/match_peaks.py -r results/Test/Test_gru_model_dpo_ta_predictions.csv
python utils/match_peaks.py -r results/Test/Test_lstm_model_predictions.csv
python utils/match_peaks.py -r results/Test/Test_lstm_model_dpo_predictions.csv
python utils/match_peaks.py -r results/Test/Test_lstm_model_ta_predictions.csv
python utils/match_peaks.py -r results/Test/Test_lstm_model_dpo_ta_predictions.csv