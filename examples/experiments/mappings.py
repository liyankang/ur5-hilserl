from experiments.ram_insertion.config import TrainConfig as RAMInsertionTrainConfig
from experiments.usb_insertion.config import TrainConfig as USBInsertionTrainConfig


CONFIG_MAPPING = {
    "ram_insertion": RAMInsertionTrainConfig,
    "usb_insertion": USBInsertionTrainConfig,
}
