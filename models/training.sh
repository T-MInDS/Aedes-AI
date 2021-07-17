set -e

#python models/training.py models/configs/ff_config.json
#python models/training.py models/configs/ff_config_dpo.json
#python models/training.py models/configs/ff_config_ta.json
#python models/training.py models/configs/ff_config_dpo_ta.json
#python models/training.py models/configs/gru_config.json
#python models/training.py models/configs/gru_config_dpo.json
#python models/training.py models/configs/gru_config_ta.json
#python models/training.py models/configs/gru_config_dpo_ta.json
#python models/training.py models/configs/lstm_config.json
#python models/training.py models/configs/lstm_config_dpo.json
#python models/training.py models/configs/lstm_config_ta.json
#python models/training.py models/configs/lstm_config_dpo_ta.json

#python models/training.py models/configs/ff_config.json
#python models/training.py models/configs/ff_config_hi.json
#python models/training.py models/configs/ff_config_lo.json
#python models/training.py models/configs/ff_config_hi_lo.json

#python models/training.py models/configs/gru_config.json
#python models/training.py models/configs/gru_config_hi.json
#python models/training.py models/configs/gru_config_lo.json
python models/training.py models/configs/gru_config_hi_lo.json

python models/training.py models/configs/lstm_config.json
python models/training.py models/configs/lstm_config_hi.json
python models/training.py models/configs/lstm_config_lo.json
python models/training.py models/configs/lstm_config_hi_lo.json

python models/training.py models/configs/lr_config.json
python models/training.py models/configs/lr_config_hi.json
python models/training.py models/configs/lr_config_lo.json
python models/training.py models/configs/lr_config_hi_lo.json