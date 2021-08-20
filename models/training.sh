set -e

python models/training.py models/configs/ff_config.json
python models/training.py models/configs/ff_config_hi.json
python models/training.py models/configs/ff_config_lo.json
python models/training.py models/configs/ff_config_hi_lo.json
python models/training.py models/configs/ff_config_2000.json

python models/training.py models/configs/gru_config.json
python models/training.py models/configs/gru_config_hi.json
python models/training.py models/configs/gru_config_lo.json
python models/training.py models/configs/gru_config_hi_lo.json
python models/training.py models/configs/gru_config_2000.json

python models/training.py models/configs/lstm_config.json
python models/training.py models/configs/lstm_config_hi.json
python models/training.py models/configs/lstm_config_lo.json
python models/training.py models/configs/lstm_config_hi_lo.json
python models/training.py models/configs/lstm_config_2000.json

python models/training.py models/configs/lr_config.json
python models/training.py models/configs/lr_config_hi.json
python models/training.py models/configs/lr_config_lo.json
python models/training.py models/configs/lr_config_hi_lo.json