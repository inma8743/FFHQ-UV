import os
import time
import torch

from RGB_Fitting.dataset.fit_dataset import FitDataset

def process_data(input_dir, output_dir):
    checkpoints_dir = '../checkpoints'
    topo_assets_dir = '../topo_assets'

    dataset_op = FitDataset(lm_detector_path='../checkpoints/lm_model/68lm_detector.pb',
                            mtcnn_detector_path='../checkpoints/mtcnn_model/mtcnn_model.pb',
                            parsing_model_pth='../checkpoints/parsing_model/79999_iter.pth',
                            parsing_resnet18_path='../checkpoints/resnet_model/resnet18-5c106cde.pth',
                            lm68_3d_path='../topo_assets/similarity_Lm3D_all.mat',
                            batch_size=1,
                            device='cuda')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir + '_vis', exist_ok=True)
    fnames = [
        fn for fn in sorted(os.listdir(input_dir))
        if fn.endswith('.jpg') or fn.endswith('.png') or fn.endswith('.jpeg')
    ]
    for fn in fnames:
        tic = time.time()
        basename = fn[:fn.rfind('.')]
        input_data = dataset_op.get_input_data(os.path.join(input_dir, fn))
        if input_data is None:
            continue

        torch.save(input_data, os.path.join(output_dir, f'{basename}.pt'))

        # input_img = tensor2np(input_data['img'][:1, :, :, :])
        # skin_img = tensor2np(input_data['skin_mask'][:1, :, :, :])
        # skin_img = img3channel(skin_img)
        # parse_mask = tensor2np(input_data['parse_mask'][:1, :, :, :], dst_range=1.0)
        # parse_img = draw_mask(input_img, parse_mask)
        # gt_lm = input_data['lm'][0, :, :].detach().cpu().numpy()
        # gt_lm[..., 1] = input_img.shape[0] - 1 - gt_lm[..., 1]
        # lm_img = draw_landmarks(input_img, gt_lm, color='b')
        # combine_img = np.concatenate([input_img, skin_img, parse_img, lm_img], axis=1)
        # save_img(combine_img, os.path.join(output_dir + '_vis', f'{basename}.png'))

        toc = time.time()
        print(f'Process image: {fn} done, took {toc - tic:.4f} seconds.')