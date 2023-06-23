import glob
import torch
import os

os.makedirs('runs/swa', exist_ok=True)

def average_checkpoints(input_ckpts, output_ckpt):
    assert len(input_ckpts) >= 1
    data = torch.load(input_ckpts[0], map_location='cpu')['state_dict']
    swa_n = 1
    for ckpt in input_ckpts[1:]:
        new_data = torch.load(ckpt, map_location='cpu')['state_dict']
        swa_n += 1
        for k, v in new_data.items():
            if v.dtype != torch.float32:
                print(k)
            else:
                data[k] += (new_data[k] - data[k]) / swa_n

    torch.save(dict(state_dict=data), output_ckpt)

WHO = 'runs/10ep_3e-5_aug_2_extracted_7'

ckpts = sorted(glob.glob(WHO + '/weights/*.ckpt'))[4:]
print(ckpts)
average_checkpoints(ckpts, 'runs/swa/10ep_3e-5_aug_2_extracted_7.ckpt')
