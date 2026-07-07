import datetime
import os


class Parameters:
    def __init__(self, result_dir='results', experiment_name=''):
        # Model parameters
        self.num_layers = 16
        self.attention_heads = 16
        self.load_weights = None  # 'model_weights/MiniLanguageModel_epoch=01.pt'
        # Training parameters
        self.do_training = True
        self.epochs = 20
        self.batch_size = 100
        self.learning_rate = 1.0
        self.lr_gamma = 0.5
        self.lr_step = 20
        self.no_gpu = False
        # Miscellaneous
        self.debug_mode = False
        self.log_interval = 1_000
        self.dry_run = False
        self.result_dir = result_dir
        self.experiment_name = experiment_name if experiment_name else datetime.datetime.now().strftime("%y%m%d-%H%M%S")

        self.save_path = os.path.join(self.result_dir, self.experiment_name)

    def save_parameters(self, file_name=None) -> None:
        """Saves the parameters to a file and writes it to the terminal.
        """
        arg_file = None
        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)
            file_name = file_name if file_name else 'parameters.txt'
            arg_file = open(os.path.join(self.save_path, file_name), 'a+')

        print("Arguments:")
        for arg in dir(self):
            if arg.startswith('_') or arg == 'save_parameters':
                continue
            line = f"- {arg}: {self.__getattribute__(arg)}"
            print(line)
            if arg_file:
                arg_file.write(line + "\n")