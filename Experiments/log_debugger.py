import torch
import logging
import os 
import sys

def build_logger(name):
    """
    Build a logger with the specified name.

    Args:
        name (str): The name of the logger.

    Returns:
        logger (logging.Logger): The logger object.
        log_dir (str): The directory path where the log file is stored.
    """
    log_dir = os.path.join('FidelityFusion_Models','log', name)
    log_file = '{}/train.log'.format(log_dir)
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)) # 重点

    sys.excepthook = handle_exception
    return logger, log_dir

class log_debugger:
    def __init__(self,method_name):
        """
        Initializes a log_debugger object.

        Parameters:
        - method_name (str): The name of the method being logged.

        Returns:
        - None
        """
        self.loss = None
        logger, _ = build_logger(method_name)
        self.logger=logger
        logger.info('torch_version:{}'.format(torch.__version__))
    
    def get_status(self,GPmodel,optimizer,epoch,loss):
        """
        Checks the status of the training process.

        Parameters:
        - GPmodel: The model being trained.
        - optimizer: The optimizer used for training.
        - epoch (int): The current epoch number.
        - loss: The loss value.

        Returns:
        - int: -1 if the loss is NaN, otherwise returns 0.
        """
        if torch.isnan(loss).any():
            self.logger.error(f"Loss is error at epoch {epoch}. Rolling back to epoch {epoch - 1}.")
        
            GPmodel.load_state_dict(self.saved_model_state)
            optimizer.load_state_dict(self.saved_optimizer_state)

            torch.save({
                'epoch': epoch - 1,
                'model_state_dict': GPmodel.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, './checkpoint/rollback_checkpoint.pth')
            
            return -1
        else:
            self.saved_model_state = GPmodel.state_dict()
            self.saved_optimizer_state = optimizer.state_dict()
            self.logger.debug(f'Epoch {epoch} - Model Parameters: {self.saved_model_state}')
            # self.logger.debug(f'Epoch {epoch} - Optimizer Parameters: {self.saved_optimizer_state}')
            
    
    