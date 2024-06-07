import yaml

from src.configs.train_config import TrainConfig, LogConfig, RenderConfig, OptimConfig, GuideConfig

from src.training.trainer import TEXTure

def load_config(yaml_path):
    with open(yaml_path, 'r') as file:
        config_data = yaml.safe_load(file)
        config = TrainConfig(
            log=LogConfig(**config_data['log']),
            render=RenderConfig(**config_data['render']),
            optim=OptimConfig(**config_data['optim']),
            guide=GuideConfig(**config_data['guide'])
        )
        return config

def render_model(eval_cfg):
    trainer = TEXTure(load_config(eval_cfg))
    trainer.full_eval()

def paint_model_w_mask(train_cfg):
    trainer = TEXTure(load_config(train_cfg))
    trainer.paint()