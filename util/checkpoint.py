import os
import torch
import logging


class CheckpointIO(object):
    def __init__(self, fname_template, data_parallel=False, **kwargs):
        os.makedirs(os.path.dirname(fname_template), exist_ok=True)
        self.fname_template = fname_template
        self.module_dict = kwargs
        self.data_parallel = data_parallel

    def register(self, **kwargs):
        self.module_dict.update(kwargs)

    def save(self, epoch):
        fname = self.fname_template.format(epoch)
        logging.info(f">> Saving checkpoint to {fname}")
        outdict = {}
        for name, module in self.module_dict.items():
            logging.info(f">> Saving {name}")
            if self.data_parallel:
                outdict[name] = module.module.state_dict()
            else:
                outdict[name] = module.state_dict()

        torch.save(outdict, fname)

    def load(self, epoch):
        fname = self.fname_template.format(epoch)
        assert os.path.exists(fname), fname + " does not exist!"
        logging.info(f">> Loading checkpoint from {fname}")
        if torch.cuda.is_available():
            module_dict = torch.load(fname)
        else:
            module_dict = torch.load(fname, map_location=torch.device("cpu"))

        for name, module in self.module_dict.items():
            if "MLPHead" in name:
                continue
            if self.data_parallel:
                module.module.load_state_dict(module_dict[name])
            else:
                module.load_state_dict(module_dict[name])
