"""funcGNN runner."""

from utils import tab_printer
from funcgnn import funcGNNTrainer
from param_parser import parameter_parser

def main():
    """
    Parsing command line parameters, reading data.
    Fitting and scoring a funcGNN model.
    """
    args = parameter_parser()
    tab_printer(args)
    trainer = funcGNNTrainer(args)
    trainer.fit()
    #trainer.start_parallel()
    #trainer.load_model()




if __name__ == "__main__":
    main()
