import argparse
import os

import torch

from src import utils
from src.lightning import DDPM
from src.linker_size_lightning import SizeClassifier
from src.visualizer import save_xyz_file
from src.datasets import (
    collate,
    collate_with_fragment_edges,
    DiffLinkerDataModule
)
from tqdm import tqdm

from pdb import set_trace


# -------------------------
# 采样脚本主体
# -------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', action='store', type=str, required=True)
parser.add_argument('--samples', action='store', type=str, required=True)
parser.add_argument('--data', action='store', type=str, required=False, default=None)
parser.add_argument('--prefix', action='store', type=str, required=True)
parser.add_argument('--n_samples', action='store', type=int, required=True)
parser.add_argument('--n_steps', action='store', type=int, required=False, default=None)
parser.add_argument('--linker_size_model', action='store', type=str, required=False, default=None)
parser.add_argument('--device', action='store', type=str, required=True)
args = parser.parse_args()

experiment_name = args.checkpoint.split('/')[-1].replace('.ckpt', '')

if args.linker_size_model is None:
    output_dir = os.path.join(args.samples, args.prefix, experiment_name)
else:
    linker_size_name = args.linker_size_model.split('/')[-1].replace('.ckpt', '')
    output_dir = os.path.join(args.samples, args.prefix, 'sampled_size', linker_size_name, experiment_name)

os.makedirs(output_dir, exist_ok=True)


def check_if_generated(_output_dir, _uuids, n_samples):
    generated = True
    starting_points = []
    for _uuid in _uuids:
        uuid_dir = os.path.join(_output_dir, _uuid)
        if not os.path.exists(uuid_dir):
            generated = False
            starting_points.append(0)
            continue

        numbers = []
        for fname in os.listdir(uuid_dir):
            try:
                num = int(fname.split('_')[0])
                numbers.append(num)
            except Exception:
                continue

        if len(numbers) == 0 or max(numbers) != n_samples - 1:
            generated = False
            if len(numbers) == 0:
                starting_points.append(0)
            else:
                # 原始代码就是 max(numbers) - 1，这里保持不变
                starting_points.append(max(numbers) - 1)

    if len(starting_points) > 0:
        starting = min(starting_points)
    else:
        starting = None

    return generated, starting


# -------------------------
# linker size 相关（如果有）
# -------------------------
collate_fn = collate
sample_fn = None
size_nn = None

if args.linker_size_model is not None:
    size_nn = SizeClassifier.load_from_checkpoint(args.linker_size_model, map_location=args.device)
    size_nn = size_nn.eval().to(args.device)

    collate_fn = collate_with_fragment_edges

    def sample_fn(_data):
        output, _ = size_nn.forward(_data)
        probabilities = torch.softmax(output, dim=1)
        distribution = torch.distributions.Categorical(probs=probabilities)
        samples = distribution.sample()
        sizes = []
        for label in samples.detach().cpu().numpy():
            sizes.append(size_nn.linker_id2size[label])
        sizes = torch.tensor(sizes, device=samples.device, dtype=torch.long)
        return sizes


# -------------------------
# 加载模型
# -------------------------
model = DDPM.load_from_checkpoint(args.checkpoint, map_location=args.device)
model = model.eval().to(args.device)
model.torch_device = args.device

# 采样阶段视作 test 阶段：用 test_data_prefix 指定要采样的数据
datamodule = DiffLinkerDataModule(
    data_path=args.data,
    train_data_prefix=None,
    val_data_prefix=None,
    test_data_prefix=args.prefix,
    batch_size=16,
    dataset_device='cpu',
    collate_fn=collate_fn,
    num_workers=4,
)

datamodule.prepare_data()
datamodule.setup(stage='test')
dataloader = datamodule.test_dataloader()
print(f'Dataloader contains {len(dataloader)} batches')

# 关键：为了兼容 DiffLinker 里写死的 isinstance(model.val_dataset, MOADDataset) 等逻辑，
# 我们把 test_dataset 显式挂到 model.val_dataset 上。
model.val_dataset = datamodule.test_dataset

# 如果想减少采样步数
if args.n_steps is not None:
    model.edm.T = args.n_steps

# -------------------------
# 采样主循环
# -------------------------
for batch_idx, data in enumerate(dataloader):
    # 把 batch 转到设备
    data = model.transfer_batch_to_device(
        data,
        device=args.device,
        dataloader_idx=0,
    )

    uuids = []
    true_names = []
    frag_names = []
    pock_names = []
    for uuid in data['uuid']:
        uuid = str(uuid)
        uuids.append(uuid)
        true_names.append(f'{uuid}/true')
        frag_names.append(f'{uuid}/frag')
        pock_names.append(f'{uuid}/pock')
        os.makedirs(os.path.join(output_dir, uuid), exist_ok=True)

    generated, starting_point = check_if_generated(output_dir, uuids, args.n_samples)
    if generated:
        print(f'Already generated batch={batch_idx}, max_uuid={max(uuids)}')
        continue
    if starting_point is None:
        starting_point = 0
    if starting_point > 0:
        print(f'Generating {args.n_samples - starting_point} for batch={batch_idx}')

    # -------------------------
    # 去中心：优先使用 fragment_only_mask / fragment_mask / anchors
    # -------------------------
    h = data['one_hot']
    x = data['positions']
    node_mask = data['atom_mask']
    frag_mask = data['fragment_mask']

    if model.inpainting:
        center_of_mass_mask = node_mask
    else:
        # 与原始逻辑保持一致，只是改成看 batch 字段而不是 model.val_dataset 类型
        if ('fragment_only_mask' in data) and model.center_of_mass == 'fragments':
            center_of_mass_mask = data['fragment_only_mask']
        elif model.center_of_mass == 'fragments':
            center_of_mass_mask = data['fragment_mask']
        elif model.center_of_mass == 'anchors':
            center_of_mass_mask = data['anchors']
        else:
            raise NotImplementedError(model.center_of_mass)

    x = utils.remove_partial_mean_with_mask(x, node_mask, center_of_mass_mask)
    utils.assert_partial_mean_zero_with_mask(x, node_mask, center_of_mass_mask)

    # -------------------------
    # 根据 batch 是否包含 pocket_mask 判断是否有 pocket
    # -------------------------
    has_pocket = 'pocket_mask' in data

    if has_pocket:
        pock_mask = data['pocket_mask']
        # ligand = atom_mask - pocket_mask
        ligand_mask = node_mask - pock_mask
        # MOAD 的 fragment_only_mask 通常是纯 fragment
        if 'fragment_only_mask' in data:
            frag_mask = data['fragment_only_mask']
        # 保存 pocket
        save_xyz_file(output_dir, h, x, pock_mask, pock_names, is_geom=model.is_geom)
    else:
        ligand_mask = node_mask
        # frag_mask 已从 data['fragment_mask'] 读取

    # 保存 ground-truth 的 ligand（不包括 pocket）
    save_xyz_file(output_dir, h, x, ligand_mask, true_names, is_geom=model.is_geom)

    # 保存 fragment
    save_xyz_file(output_dir, h, x, frag_mask, frag_names, is_geom=model.is_geom)

    # -------------------------
    # 采样并保存预测结果（同样剔除 pocket）
    # -------------------------
    for i in tqdm(range(starting_point, args.n_samples), desc=str(batch_idx)):
        chain, gen_node_mask = model.sample_chain(data, sample_fn=sample_fn, keep_frames=1)
        gen_x = chain[0][:, :, :model.n_dims]
        gen_h = chain[0][:, :, model.n_dims:]

        if has_pocket:
            gen_node_mask = gen_node_mask - data['pocket_mask']

        pred_names = [f'{uuid}/{i}' for uuid in uuids]
        save_xyz_file(output_dir, gen_h, gen_x, gen_node_mask, pred_names, is_geom=model.is_geom)
