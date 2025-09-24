from exc.train_noTX import Train_noTX
from config.base_config import BaseConfig




if __name__ == '__main__':
   cfg = BaseConfig('./config/noTX.toml', {})
   print(cfg)
   trainer = Train_noTX(cfg)
   trainer.train()
