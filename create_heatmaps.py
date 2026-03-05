from __future__ import print_function

import numpy as np

import argparse

import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from math import floor
from models.TransMIL import Transmil
import h5py
import yaml
from wsi_core.batch_process_utils import initialize_df
from vis_utils.heatmap_utils import initialize_wsi, drawHeatmap, compute_from_patches
from utils.file_utils import save_hdf5
from utils.eval_utils import initiate_model as initiate_model


import glob

parser = argparse.ArgumentParser(description='Heatmap inference script')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code')
parser.add_argument('--overlap', type=float, default=None)
parser.add_argument('--config_file', type=str, default="heatmap_config_template.yaml")
args = parser.parse_args()

def infer_single_slide(model, features, label, reverse_label_dict, k=1):
    features = features.to(device)
    with torch.no_grad():
        if isinstance(model, (Transmil)):
            model = model.to('cuda')
            features = features.to('cuda')
            logits, Y_prob, Y_hat, A_raw, _ = model(data=features)	
            print(A_raw.shape)
            B, heads, H, W = A_raw.shape 
            n = features.shape[0]  
            
            A_raw_without_cls = A_raw[:, :, 1:n + 1, 1:n + 1] 
            A_mean = A_raw_without_cls.mean(dim=1)  
            
            A_max = A_mean.mean(dim=2)[0]  # Maximum attention per instance
            A_max = A_max.squeeze(0).cpu().numpy()
            

        print('Y: {}, Y_prob: {}'.format(label, ["{:.4f}".format(p) for p in Y_prob.cpu().flatten()]))	
        probs, ids = torch.topk(Y_prob, k)
        probs = probs[-1].cpu().numpy()
        ids = ids[-1].cpu().numpy()
        preds_str = np.array([reverse_label_dict[idx] for idx in ids])


    return ids, preds_str, probs, A_max

def load_params(df_entry, params):
    for key in params.keys():
        if key in df_entry.index:
            dtype = type(params[key])
            val = df_entry[key] 
            val = dtype(val)
            if isinstance(val, str):
                if len(val) > 0:
                    params[key] = val
            elif not np.isnan(val):
                params[key] = val
            else:
                pdb.set_trace()

    return params

def parse_config_dict(args, config_dict):
    if args.save_exp_code is not None:
        config_dict['exp_arguments']['save_exp_code'] = args.save_exp_code
    if args.overlap is not None:
        config_dict['patching_arguments']['overlap'] = args.overlap
    return config_dict

if __name__ == '__main__':
    config_path = os.path.join('heatmaps/configs', args.config_file)
    config_dict = yaml.safe_load(open(config_path, 'r'))
    config_dict = parse_config_dict(args, config_dict)

    for key, value in config_dict.items():
        if isinstance(value, dict):
            print('\n'+key)
            for value_key, value_value in value.items():
                print (value_key + " : " + str(value_value))
        else:
            print ('\n'+key + " : " + str(value))
            
    model_path = "config/TransMIL.yaml"
    model_dict = yaml.safe_load(open(model_path, 'r'))
    model_args = parse_config_dict(args, model_dict)
    
    decision = 'y'	
    #decision = input('Continue? Y/N ')
    if decision in ['Y', 'y', 'Yes', 'yes']:
        pass
    elif decision in ['N', 'n', 'No', 'NO']:
        exit()
    else:
        raise NotImplementedError

    args = config_dict
    patch_args = argparse.Namespace(**args['patching_arguments'])
    data_args = argparse.Namespace(**args['data_arguments'])
    m_args = argparse.Namespace(**args['model_arguments'])

    
    exp_args = argparse.Namespace(**args['exp_arguments'])
    heatmap_args = argparse.Namespace(**args['heatmap_arguments'])
    sample_args = argparse.Namespace(**args['sample_arguments'])

    patch_size = tuple([patch_args.patch_size for i in range(2)])
    step_size = tuple((np.array(patch_size) * (1-patch_args.overlap)).astype(int))
    print('patch_size: {} x {}, with {:.2f} overlap, step size is {} x {}'.format(patch_size[0], patch_size[1], patch_args.overlap, step_size[0], step_size[1]))

    
    preset = data_args.preset
    def_seg_params = {'seg_level': -1, 'sthresh': 15, 'mthresh': 11, 'close': 2, 'use_otsu': False, 
                      'keep_ids': 'none', 'exclude_ids':'none'}
    def_filter_params = {'a_t':50.0, 'a_h': 8.0, 'max_n_holes':10}
    def_vis_params = {'vis_level': -1, 'line_thickness': 250}
    def_patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

    if preset is not None:
        preset_df = pd.read_csv(preset)
        for key in def_seg_params.keys():
            def_seg_params[key] = preset_df.loc[0, key]

        for key in def_filter_params.keys():
            def_filter_params[key] = preset_df.loc[0, key]

        for key in def_vis_params.keys():
            def_vis_params[key] = preset_df.loc[0, key]

        for key in def_patch_params.keys():
            def_patch_params[key] = preset_df.loc[0, key]


    df = pd.read_csv(os.path.join('heatmaps/process_lists', data_args.process_list))
    df = initialize_df(df, def_seg_params, def_filter_params, def_vis_params, def_patch_params, use_heatmap_args=False)

    mask = df['process'] == 1
    process_stack = df[mask].reset_index(drop=True)
    total = len(process_stack)
    print('\nlist of slides to process: ')
    print(process_stack.head(len(process_stack)))

    print('\ninitializing model from checkpoint')
    #ckpt_path = model_args.ckpt_path
    ckpt_path = m_args.ckpt_path
    print('\nckpt path: {}'.format(ckpt_path))
    
    model =  initiate_model(model_args, ckpt_path)

  
    
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Done!')

    label_dict =  data_args.label_dict
    class_labels = list(label_dict.keys())
    class_encodings = list(label_dict.values())
    reverse_label_dict = {class_encodings[i]: class_labels[i] for i in range(len(class_labels))} 

    if torch.cuda.device_count() > 1:
        device_ids = list(range(torch.cuda.device_count()))

    os.makedirs(exp_args.production_save_dir, exist_ok=True)
    os.makedirs(exp_args.raw_save_dir, exist_ok=True)
    blocky_wsi_kwargs = {'top_left': None, 'bot_right': None, 'patch_size': patch_size, 'step_size': patch_size, 
    'custom_downsample':patch_args.custom_downsample, 'level': patch_args.patch_level, 'use_center_shift': heatmap_args.use_center_shift}

    for i in range(len(process_stack)):
        slide_name = process_stack.loc[i, 'slide_id']
        if data_args.slide_ext not in slide_name:
            slide_name+=data_args.slide_ext
        print('\nprocessing: ', slide_name)	

        try:
            label = process_stack.loc[i, 'label']
        except KeyError:
            label = 'Unspecified'

        slide_id = slide_name.replace(data_args.slide_ext, '')

        if not isinstance(label, str):
            grouping = reverse_label_dict[label]
        else:
            grouping = label


        r_slide_save_dir = "./heatmaps/heatmap_raw_results/"
        p_slide_save_dir = "./heatmaps/heatmap_raw_results/"

        if heatmap_args.use_roi:
            x1, x2 = process_stack.loc[i, 'x1'], process_stack.loc[i, 'x2']
            y1, y2 = process_stack.loc[i, 'y1'], process_stack.loc[i, 'y2']
            top_left = (int(x1), int(y1))
            bot_right = (int(x2), int(y2))
        else:
            top_left = None
            bot_right = None
        
        print('slide id: ', slide_id)
        print('top left: ', top_left, ' bot right: ', bot_right)

        slide_path = glob.glob("/data_directory/"+ slide_name + "*")[0]
                

        
        mask_file = os.path.join(r_slide_save_dir, slide_id+'_mask.pkl')
        
        # Load segmentation and filter parameters
        seg_params = def_seg_params.copy()
        filter_params = def_filter_params.copy()
        vis_params = def_vis_params.copy()

        seg_params = load_params(process_stack.loc[i], seg_params)
        filter_params = load_params(process_stack.loc[i], filter_params)
        vis_params = load_params(process_stack.loc[i], vis_params)

        keep_ids = str(seg_params['keep_ids'])
        if len(keep_ids) > 0 and keep_ids != 'none':
            seg_params['keep_ids'] = np.array(keep_ids.split(',')).astype(int)
        else:
            seg_params['keep_ids'] = []

        exclude_ids = str(seg_params['exclude_ids'])
        if len(exclude_ids) > 0 and exclude_ids != 'none':
            seg_params['exclude_ids'] = np.array(exclude_ids.split(',')).astype(int)
        else:
            seg_params['exclude_ids'] = []

        for key, val in seg_params.items():
            print('{}: {}'.format(key, val))

        for key, val in filter_params.items():
            print('{}: {}'.format(key, val))

        for key, val in vis_params.items():
            print('{}: {}'.format(key, val))
        
        print('Initializing WSI object')
        wsi_object = initialize_wsi(slide_path, seg_mask_path=mask_file, seg_params=seg_params, filter_params=filter_params)
        print('Done!')

        wsi_ref_downsample = wsi_object.level_downsamples[patch_args.patch_level]

        # the actual patch size for heatmap visualization should be the patch size * downsample factor * custom downsample factor
        vis_patch_size = tuple((np.array(patch_size) * np.array(wsi_ref_downsample) * patch_args.custom_downsample).astype(int))

        block_map_save_path = os.path.join(r_slide_save_dir, '{}_blockmap.h5'.format(slide_id))
        mask_path = os.path.join(r_slide_save_dir, '{}_mask.jpg'.format(slide_id))
        if vis_params['vis_level'] < 0:
            best_level = wsi_object.wsi.get_best_level_for_downsample(32)
            vis_params['vis_level'] = best_level
        mask = wsi_object.visWSI(**vis_params, number_contours=True)
        mask.save(mask_path)
        
        features_path = os.path.join(r_slide_save_dir, slide_id+'.pt')
        h5_path = os.path.join(r_slide_save_dir, slide_id+'.h5')
        
        features = torch.load(features_path)
        print("features: ", features.shape)
        process_stack.loc[i, 'bag_size'] = len(features)
        
        wsi_object.saveSegmentation(mask_file)
        Y_hats, Y_hats_str, Y_probs, A = infer_single_slide(model, features, label, reverse_label_dict, exp_args.n_classes)
        del features
        if not os.path.isfile(block_map_save_path): 
            file = h5py.File(h5_path, "r")
            coords = file['coords'][:]
            file.close()
            asset_dict = {'attention_scores': A, 'coords': coords}
            print("A, coords: ", A.shape, coords.shape)
            block_map_save_path = save_hdf5(block_map_save_path, asset_dict, mode='w')
        
        # save top 3 predictions
        for c in range(exp_args.n_classes):
            process_stack.loc[i, 'Pred_{}'.format(c)] = Y_hats_str[c]
            process_stack.loc[i, 'p_{}'.format(c)] = Y_probs[c]

        os.makedirs('heatmaps/results/', exist_ok=True)
        if data_args.process_list is not None:
            process_stack.to_csv('heatmaps/results/{}.csv'.format(data_args.process_list.replace('.csv', '')), index=False)
        else:
            process_stack.to_csv('heatmaps/results/{}.csv'.format(exp_args.save_exp_code), index=False)

        
        file = h5py.File(block_map_save_path, 'r')
        print(list(file.keys()))
        


        wsi_kwargs = {'top_left': top_left, 'bot_right': bot_right, 'patch_size': patch_size, 'step_size': step_size, 
        'custom_downsample':patch_args.custom_downsample, 'level': patch_args.patch_level, 'use_center_shift': heatmap_args.use_center_shift}

        heatmap_save_name = '{}_blockmap.tiff'.format(slide_id)
        if os.path.isfile(os.path.join(r_slide_save_dir, heatmap_save_name)):
            pass
        else:   
            heatmap = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object, cmap=heatmap_args.cmap, alpha=heatmap_args.alpha, use_holes=True, binarize=False, vis_level=-1, blank_canvas=False,
                            thresh=-1, patch_size = vis_patch_size, convert_to_percentiles=True)
        
            heatmap.save(os.path.join(r_slide_save_dir, '{}_blockmap.png'.format(slide_id)))
            del heatmap

        save_path = os.path.join(r_slide_save_dir, '{}_{}_roi_{}.h5'.format(slide_id, patch_args.overlap, heatmap_args.use_roi))

        if heatmap_args.use_ref_scores:
            ref_scores = scores
        else:
            ref_scores = None
        
        if heatmap_args.calc_heatmap:
            compute_from_patches(wsi_object=wsi_object, clam_pred=Y_hats[0], model=model, batch_size=exp_args.batch_size, **wsi_kwargs, 
                                attn_save_path=save_path,  ref_scores=ref_scores)

        if not os.path.isfile(save_path):
            print('heatmap {} not found'.format(save_path))
            if heatmap_args.use_roi:
                save_path_full = os.path.join(r_slide_save_dir, '{}_{}_roi_False.h5'.format(slide_id, patch_args.overlap))
                print('found heatmap for whole slide')
                save_path = save_path_full
            else:
                continue

        file = h5py.File(save_path, 'r')
        dset = file['attention_scores']
        coord_dset = file['coords']
        scores = dset[:]
        coords = coord_dset[:]


        file.close()

       