from pathlib import Path
from typing import Any, Dict, Union, List

import cv2
import imageio
import numpy as np
import pyrallis
import torch
import torch.nn.functional as F
from PIL import Image
from loguru import logger
from matplotlib import cm
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import os

from src.configs.train_config import TrainConfig
from src.models.textured_mesh import TexturedMeshModel
from src.training.views_dataset import ViewsDataset, MultiviewDataset
from src.utils import make_path, tensor2numpy, seed_everything, gaussian_blur, color_with_shade


class TEXTure:
    # from https://github.com/TEXTurePaper/TEXTurePaper
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.paint_step = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        seed_everything(self.cfg.optim.seed)

        # Make view_dirs
        self.exp_path = make_path(self.cfg.log.exp_dir) 
        if self.cfg.log.eval_only:
            self.final_renders_path = make_path(self.exp_path / 'results')
        else:
            self.mask_path = make_path(self.exp_path / 'masks')

        self.init_logger()
        pyrallis.dump(self.cfg, (self.exp_path / 'config.yaml').open('w'))

        # load mesh
        self.mesh_model = self.init_mesh_model()
        self.dataloaders = self.init_dataloaders()

        logger.info(f'Successfully initialized {self.cfg.log.exp_name}')

    def init_mesh_model(self) -> nn.Module:
        cache_path = None
        model = TexturedMeshModel(self.cfg.guide, device=self.device,
                                  render_grid_size=self.cfg.render.train_grid_size,
                                  cache_path=cache_path,
                                  texture_resolution=self.cfg.guide.texture_resolution,
                                  augmentations=False)

        model = model.to(self.device)
        logger.info(
            f'Loaded Mesh, #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')
        logger.info(model)
        return model

    def init_dataloaders(self) -> Dict[str, DataLoader]:
        init_train_dataloader = MultiviewDataset(self.cfg.render, device=self.device).dataloader()

        val_loader = ViewsDataset(self.cfg.render, device=self.device,
                                  size=self.cfg.log.eval_size).dataloader()
        # Will be used for creating the final video
        val_large_loader = ViewsDataset(self.cfg.render, device=self.device,
                                        size=self.cfg.log.full_eval_size).dataloader()
        dataloaders = {'train': init_train_dataloader, 'val': val_loader,
                       'val_large': val_large_loader}
        return dataloaders

    def init_logger(self):
        logger.remove()  # Remove default logger
        log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{message}</level>"
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, format=log_format)
        logger.add(self.exp_path / 'log.txt', colorize=False, format=log_format)

    def paint(self):
        logger.info('Starting training ^_^')
        
        # Evaluate the initialization
        self.mesh_model.train()

        mat2im_path = self.cfg.guide.mat2im_path
        mat2im = json.load(open(mat2im_path, 'r'))
        
        result_path = self.cfg.guide.result_path

        sel_mat = ""
        sel_im_list = []

        pbar = tqdm(total=len(mat2im), initial=self.paint_step,
                    bar_format='{desc}: {percentage:3.0f}% painting step {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        for material, im_list in mat2im.items():
            self.mesh_model = self.init_mesh_model()
            self.paint_step = 0
            sel_mat = material
            sel_im_list = im_list

            pbar.update(1)
            mask_im_dic = {}
            for im_str in sel_im_list:
                ori_im, part_num = im_str.split(', ')
                view_num = ori_im.split('/')[-2]
                mask_im = ori_im.replace('full.jpg', f'{part_num}_mask.png')
                if view_num not in mask_im_dic:
                    mask_im_dic[view_num] = [mask_im]
                else:
                    mask_im_dic[view_num].append(mask_im)

            for idx, data in enumerate(self.dataloaders['train']):
                self.paint_step += 1
                if str(idx) not in mask_im_dic.keys():
                    continue
                mask_im_list = mask_im_dic[str(idx)]
                for mask_path in mask_im_list:
                    if not os.path.exists(mask_path): continue
                    mask = np.array(Image.open(mask_path))
                    mask = torch.tensor(mask/255.).float().to(self.device)[None, :, :, None]
                    output_path = f'{result_path}/step_00000_{self.paint_step-1:04d}_rgb.jpg' 
                    output = np.array(Image.open(output_path))
                    output = torch.tensor(output.transpose((2,0,1))[None,:]/255.).float().to(self.device)
                    self.paint_viewpoint(data, pre_mask=mask, pre_output=output)
                    self.mesh_model.train()

            self.mesh_model.change_default_to_median(save_name=f'{self.mask_path}/{sel_mat}.png')
            logger.info('Finished Painting ^_^')
            logger.info('Saving the last result...')
            logger.info('\tDone!')

    def evaluate(self, dataloader: DataLoader, save_path: Path, save_as_video: bool = False): 
        logger.info(f'Evaluating and saving model, painting iteration #{self.paint_step}...')
        self.mesh_model.eval()
        save_path.mkdir(exist_ok=True)

        if save_as_video:
            all_preds = []
        for i, data in enumerate(dataloader):
            preds = self.eval_render(data) 
            pred = tensor2numpy(preds[0])

            if save_as_video:
                all_preds.append(pred)
            else:
                Image.fromarray(pred).save(save_path / f"step_{self.paint_step:05d}_{i:04d}_rgb.jpg")

        if save_as_video:
            all_preds = np.stack(all_preds, axis=0)

            dump_vid = lambda video, name: imageio.mimsave(save_path / f"step_{self.paint_step:05d}_{name}.mp4", video,
                                                           fps=25,
                                                           quality=8, macro_block_size=1)

            dump_vid(all_preds, 'rgb')
        logger.info('Done!')

    def full_eval(self, output_dir: Path = None):
        if output_dir is None:
            output_dir = self.final_renders_path
        self.evaluate(self.dataloaders['train'], output_dir, save_as_video=False)

        if self.cfg.log.save_mesh:
            save_path = make_path(self.exp_path / 'mesh')
            logger.info(f"Saving mesh to {save_path}")

            self.mesh_model.export_mesh(save_path)

            logger.info(f"\tDone!")

    def paint_viewpoint(self, data: Dict[str, Any], pre_mask=None, pre_output=None):
        logger.info(f'--- Painting step #{self.paint_step} ---')
        theta, phi, radius = data['theta'], data['phi'], data['radius']
        # If offset of phi was set from code
        phi = phi - np.deg2rad(self.cfg.render.front_offset)
        phi = float(phi + 2 * np.pi if phi < 0 else phi)
        logger.info(f'Painting from theta: {theta}, phi: {phi}, radius: {radius}')

        background = torch.Tensor([0, 0.8, 0]).to(self.device)

        # Render from viewpoint
        outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius, background=background, pre_mask=pre_mask) 
        render_cache = outputs['render_cache']
        rgb_render_raw = outputs['image']  
        depth_render = outputs['depth']
        # Render meta texture map
        meta_output = self.mesh_model.render(background=torch.Tensor([0, 0, 0]).to(self.device),
                                             use_meta_texture=True, render_cache=render_cache, pre_mask=pre_mask)

        z_normals = outputs['normals'][:, -1:, :, :].clamp(0, 1)
        z_normals_cache = meta_output['image'].clamp(0, 1)
        edited_mask = meta_output['image'].clamp(0, 1)[:, 1:2]

        update_mask, _, _ = self.calculate_trimap(rgb_render_raw=rgb_render_raw,
                                                                        depth_render=depth_render,
                                                                        z_normals=z_normals,
                                                                        z_normals_cache=z_normals_cache,
                                                                        edited_mask=edited_mask,
                                                                        mask=outputs['mask'])

        update_ratio = float(update_mask.sum() / (update_mask.shape[2] * update_mask.shape[3]))
        if self.cfg.guide.reference_texture is not None and update_ratio < 0.01:
            logger.info(f'Update ratio {update_ratio:.5f} is small for an editing step, skipping')
            return

        # Project back
        rgb_output = pre_output
        object_mask = outputs['mask']
        self.project_back(render_cache=render_cache, background=background, rgb_output=rgb_output,
                                               object_mask=object_mask, update_mask=update_mask, z_normals=z_normals,
                                               z_normals_cache=z_normals_cache)


    def eval_render(self, data):
        theta = data['theta']
        phi = data['phi']
        radius = data['radius']
        phi = phi - np.deg2rad(self.cfg.render.front_offset)
        phi = float(phi + 2 * np.pi if phi < 0 else phi)
        dim = self.cfg.render.eval_grid_size
        outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius,
                                         dims=(dim, dim), background='white')
        z_normals = outputs['normals'][:, -1:, :, :].clamp(0, 1)
        rgb_render = outputs['image']  # .permute(0, 2, 3, 1).contiguous().clamp(0, 1)
        render_cache = outputs['render_cache']
        face_idx = render_cache['face_idx']
        if rgb_render.shape[1] == 4:
            rgb_render = rgb_render[:, :3, :, :]
        diff = (rgb_render.detach() - torch.tensor(self.mesh_model.default_color).view(1, 3, 1, 1).to(
            self.device)).abs().sum(axis=1)
        uncolored_mask = (diff < 0.1).float().unsqueeze(0) 
        rgb_render = rgb_render * (1 - uncolored_mask) + color_with_shade([0.85, 0.85, 0.85], z_normals=z_normals,
                                                                                light_coef=0.3) * uncolored_mask

        rgb_render = rgb_render.permute(0, 2, 3, 1).contiguous().clamp(0, 1).detach()
        return rgb_render


    def calculate_trimap(self, rgb_render_raw: torch.Tensor,
                         depth_render: torch.Tensor,
                         z_normals: torch.Tensor, z_normals_cache: torch.Tensor, edited_mask: torch.Tensor,
                         mask: torch.Tensor):
        diff = (rgb_render_raw.detach() - torch.tensor(self.mesh_model.default_color).view(1, 3, 1, 1).to(
            self.device)).abs().sum(axis=1) 
        exact_generate_mask = (diff < 0.1).float().unsqueeze(0) 

        # Extend mask
        generate_mask = torch.from_numpy(
            cv2.dilate(exact_generate_mask[0, 0].detach().cpu().numpy(), np.ones((19, 19), np.uint8))).to(
            exact_generate_mask.device).unsqueeze(0).unsqueeze(0)

        update_mask = generate_mask.clone() #

        object_mask = torch.ones_like(update_mask)
        object_mask[depth_render == 0] = 0
        object_mask = torch.from_numpy(
            cv2.erode(object_mask[0, 0].detach().cpu().numpy(), np.ones((7, 7), np.uint8))).to(
            object_mask.device).unsqueeze(0).unsqueeze(0)

        # Generate the refine mask based on the z normals, and the edited mask

        refine_mask = torch.zeros_like(update_mask)
        refine_mask[z_normals > z_normals_cache[:, :1, :, :] + self.cfg.guide.z_update_thr] = 1 
        if self.cfg.guide.initial_texture is None:
            refine_mask[z_normals_cache[:, :1, :, :] == 0] = 0
        elif self.cfg.guide.reference_texture is not None:
            refine_mask[edited_mask == 0] = 0
            refine_mask = torch.from_numpy(
                cv2.dilate(refine_mask[0, 0].detach().cpu().numpy(), np.ones((31, 31), np.uint8))).to(
                mask.device).unsqueeze(0).unsqueeze(0)
            refine_mask[mask == 0] = 0
            # Don't use bad angles here
            refine_mask[z_normals < 0.4] = 0
        else:
            # Update all regions inside the object
            refine_mask[mask == 0] = 0

        refine_mask = torch.from_numpy(
            cv2.erode(refine_mask[0, 0].detach().cpu().numpy(), np.ones((5, 5), np.uint8))).to(
            mask.device).unsqueeze(0).unsqueeze(0)
        refine_mask = torch.from_numpy(
            cv2.dilate(refine_mask[0, 0].detach().cpu().numpy(), np.ones((5, 5), np.uint8))).to(
            mask.device).unsqueeze(0).unsqueeze(0)
        update_mask[refine_mask == 1] = 1 # update = refine + generate mask

        update_mask[torch.bitwise_and(object_mask == 0, generate_mask == 0)] = 0

        return update_mask, generate_mask, refine_mask

    def project_back(self, render_cache: Dict[str, Any], background: Any, rgb_output: torch.Tensor,
                     object_mask: torch.Tensor, update_mask: torch.Tensor, z_normals: torch.Tensor,
                     z_normals_cache: torch.Tensor):
        object_mask = torch.from_numpy(
            cv2.erode(object_mask[0, 0].detach().cpu().numpy(), np.ones((5, 5), np.uint8))).to(
            object_mask.device).unsqueeze(0).unsqueeze(0) 
        render_update_mask = object_mask.clone()

        render_update_mask[update_mask == 0] = 0

        blurred_render_update_mask = torch.from_numpy(
            cv2.dilate(render_update_mask[0, 0].detach().cpu().numpy(), np.ones((25, 25), np.uint8))).to(
            render_update_mask.device).unsqueeze(0).unsqueeze(0)
        blurred_render_update_mask = gaussian_blur(blurred_render_update_mask, 21, 16)

        # Do not get out of the object
        blurred_render_update_mask[object_mask == 0] = 0

        if self.cfg.guide.strict_projection:
            blurred_render_update_mask[blurred_render_update_mask < 0.5] = 0
            # Do not use bad normals
            z_was_better = z_normals + self.cfg.guide.z_update_thr < z_normals_cache[:, :1, :, :]
            blurred_render_update_mask[z_was_better] = 0

        render_update_mask = blurred_render_update_mask

        # Update the normals
        z_normals_cache[:, 0, :, :] = torch.max(z_normals_cache[:, 0, :, :], z_normals[:, 0, :, :])

        optimizer = torch.optim.Adam(self.mesh_model.get_params(), lr=self.cfg.optim.lr, betas=(0.9, 0.99),
                                     eps=1e-15) 
        for _ in tqdm(range(200), desc='fitting mesh colors'):
            optimizer.zero_grad()
            outputs = self.mesh_model.render(background=background,
                                             render_cache=render_cache) 
            rgb_render = outputs['image']

            mask = render_update_mask.flatten()
            masked_pred = rgb_render.reshape(1, rgb_render.shape[1], -1)[:, :, mask > 0]
            masked_target = rgb_output.reshape(1, rgb_output.shape[1], -1)[:, :, mask > 0] 
            masked_mask = mask[mask > 0]
            loss = ((masked_pred - masked_target.detach()).pow(2) * masked_mask).mean()

            meta_outputs = self.mesh_model.render(background=torch.Tensor([0, 0, 0]).to(self.device),
                                                  use_meta_texture=True, render_cache=render_cache)
            current_z_normals = meta_outputs['image']
            current_z_mask = meta_outputs['mask'].flatten()
            masked_current_z_normals = current_z_normals.reshape(1, current_z_normals.shape[1], -1)[:, :,
                                       current_z_mask == 1][:, :1]
            masked_last_z_normals = z_normals_cache.reshape(1, z_normals_cache.shape[1], -1)[:, :,
                                    current_z_mask == 1][:, :1]
            loss += (masked_current_z_normals - masked_last_z_normals.detach()).pow(2).mean()
            loss.backward()
            optimizer.step()

        return rgb_render, current_z_normals
