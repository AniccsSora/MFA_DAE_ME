from torch.utils.data import DataLoader

from dataset.HLsep_dataloader import hl_dataloader
from typing import List, Dict, Any, AnyStr





def get_model_input_dataloader(audio_list: List,
                               shuffle: bool,
                               num_workers: int,
                               pin_memory: bool,
                               FFT_dict: Dict,
                               args) -> List[DataLoader]:
    """

    Args:
        audio_list: 音訊位置
        shuffle:  整張FFT 是否要亂序。
        num_workers: windows OS 下請設定 0。
        pin_memory: 電腦記憶體很夠 就設定 True。(增加資料轉移效能)
        FFT_dict (object) : FFT 參數
        args: 全域參數(訓練參數)

    Returns:

    """
    res: List[DataLoader[Any]] = []
    for path in audio_list:
        res.append(hl_dataloader(audio_list,
                                 batch_size=args.batch_size,
                                 shuffle=shuffle,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory,
                                 FFT_dict=FFT_dict,
                                 args=args)
                   )
        
    return res