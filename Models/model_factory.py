import importlib

class ModelFactory:
    @staticmethod
    def get_model(model_name, model_path, model_config, out_score_file_name):
        """Return an instance of a model based on model_name."""
        module = importlib.import_module(f'Models.{model_name}')

        if model_config is not None:
            model = module.load_model(model_path, model_config, out_score_file_name)
        else:
            model = module.load_model(model_path, out_score_file_name)

        model.eval()
        
        return model

