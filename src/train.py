from config.train import TrainConfig
from src.utils.helper import run
from src.utils.trainEUC import train1
# from src.utils.trainEUCgai import train1
# from src.utils.train_normal import train1 
# from src.utils.train_ATres50 import train1 
# from src.utils.trainPCCD import train1
# from src.utils.trainPCCTrades import train1
# from src.utils.trainPCCConSmooth import train1
if __name__ == '__main__':
    config = TrainConfig()
    run(train1, config)
