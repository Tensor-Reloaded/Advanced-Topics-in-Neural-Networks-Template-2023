import logging

from torch import jit
from enum import Enum


class SaveOption(Enum):
    SCRIPT = 1
    TRACE = 2
    ALL = 3


class ModelManager:
    def __init__(self, model, trace_path="model.trace.pt", script_path="model.script.pt"):
        self.model = model
        self.trace_path = trace_path
        self.script_path = script_path
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(ModelManager.__name__)

    def trace(self, input):
        try:
            self.logger.info(f"[{ModelManager.__name__}] Started tracing...")
            self.traced_model = jit.trace(self.model, input)
            self.logger.error(f"[{ModelManager.__name__}] Finished tracing")
        except Exception as e:
            self.logger.error(f"[{ModelManager.__name__}] Something wrong happened while tracing: {e}")

    def script_model(self):
        try:
            self.logger.info(f"[{ModelManager.__name__}] Started scripting...")
            self.scripted_model = jit.script(self.model)
            self.logger.error(f"[{ModelManager.__name__}] Finished scripting")
        except Exception as e:
            self.logger.error(f"[{ModelManager.__name__}] Something wrong happened while scripting: {e}")

    def save(self, save_option=SaveOption.ALL):
        models = [('traced_model', self.trace_path), ('scripted_model', self.script_path)]
        if save_option == SaveOption.TRACE:
            models = [('traced_model', self.trace_path)]
        elif save_option == SaveOption.SCRIPT:
            models = [('scripted_model', self.script_path)]

        for model_type, model_path in models:
            if not hasattr(self, model_type):
                self.logger.warning(f"Model of type {model_type} not found")
                continue
            self.logger.info(f"Saving model of type {model_type}")
            getattr(self, model_type).save(model_path)
