import sys, json, tempfile, pathlib, importlib.util
import numpy as np
import torch
sys.path.insert(0, str(pathlib.Path('src').resolve()))
from sparse_pod_sim2real.data.dataset import SparseTrajectoryDataset
from sparse_pod_sim2real.data.sensor_topology import get_sensor_placement
from sparse_pod_sim2real.model.our_model.misf_no import PhysicsCrossAttention
from sparse_pod_sim2real.model.our_model.subspace_warping import GrassmannSubspaceAlignment
from sparse_pod_sim2real.model.our_model.gappy_solver import DifferentiableGappySolver
from sparse_pod_sim2real.model import load_model
import yaml

torch.set_num_threads(4)
torch.manual_seed(20260917)
results = {'scope':'CPU diagnostic only; no training; real-frame results are exploratory, not held-out benchmark scores','torch':torch.__version__}
# Exercise the exact split method without loading trajectories.
with tempfile.TemporaryDirectory() as td:
    root=pathlib.Path(td)
    for name in ['a.h5.pt','b.h5.pt','c.h5.pt']:(root/name).touch()
    for mode in ['train','val','test']:(root/f'{mode}_index_numerical.json').write_text(json.dumps(['missing.h5']))
    splits={}
    for mode in ['train','val','test']:
        ds=SparseTrajectoryDataset.__new__(SparseTrajectoryDataset)
        ds.tensor_dir=root;ds.data_root=root;ds.dataset_type='numerical';ds.mode=mode;ds.test_mode='all';ds.seed=42;ds.few_shot_k=None
        splits[mode]=[p.name for p in ds._split_trajectories(.5,None)]
    results['unmatched_index_splits']=splits
root=pathlib.Path(r'D:/hj/Y-3 S-1/POD-Sim2Real/data/tensor_cache_64x128')
actual={}
for mode in ['train','val','test']:
    ds=SparseTrajectoryDataset.__new__(SparseTrajectoryDataset)
    ds.tensor_dir=root/'numerical';ds.data_root=root;ds.dataset_type='numerical';ds.mode=mode;ds.test_mode='all';ds.seed=42;ds.few_shot_k=None
    actual[mode]=[p.name for p in ds._split_trajectories(.5,None)]
results['actual_local_sim_splits']=actual
attn=PhysicsCrossAttention(k=4,d_model=8,n_heads=2)
try:
    attn(torch.randn(2,3,5,2),torch.randn(2,5,2))
    results['batched_coords']='unexpected success'
except Exception as e:results['batched_coords']=str(e)
payload=torch.load('artifacts/pod_basis_64x128.pt',map_location='cpu',weights_only=False)
phi=payload['basis'];perp=payload['basis_perp'];mean=payload['mean'];k=phi.shape[1]
warp=GrassmannSubspaceAlignment(k=k,r=perp.shape[1],phi_sim=phi,phi_perp=perp)
q=warp.get_adapted_basis()
results['actual_basis_qr']={'relative_basis_change_at_A0':float(torch.linalg.norm(q-phi)/torch.linalg.norm(phi)), 'negative_column_dot_products':int(((q*phi).sum(0)<0).sum()), 'projector_distance_proxy':float(torch.linalg.norm(q-phi@(phi.T@q)))}
# Control with a valid orthonormal basis whose column signs differ from torch QR convention.
phi_control=phi.clone();phi_control[:,0]*=-1
q_control=GrassmannSubspaceAlignment(k=k,r=perp.shape[1],phi_sim=phi_control,phi_perp=perp).get_adapted_basis()
results['qr_sign_control_relative_change']=float(torch.linalg.norm(q_control-phi_control)/torch.linalg.norm(phi_control))
results['basis_manifest']=payload.get('manifest',{})
# Same 20 frames used by the existing visualization, for diagnostic decomposition only.
raw=torch.load(root/'real/10000_15.0.h5.pt',map_location='cpu',weights_only=False)
field=raw[50:70].permute(0,2,3,1).contiguous().unsqueeze(0)
results['reconstruction_probe']={}
for top in ['wall','uniform','wake_rake','random']:
    op=get_sensor_placement(top,64)
    solver=DifferentiableGappySolver(phi,op.indices_1d,mean,h=64,w=128,reg_lambda=1e-4)
    vals=op.extract_sensor_values(field)
    with torch.no_grad():
        _,recon=solver(vals);_,oracle=solver.project_full_field(field)
    sv=torch.linalg.svdvals(solver.phi_p)
    def rel(a,b):return float(torch.linalg.norm(a-b)/torch.linalg.norm(b))
    results['reconstruction_probe'][top]={'rel_l2':rel(recon,field),'oracle_projection_rel_l2':rel(oracle,field),'v_rel_l2':rel(recon[...,1],field[...,1]),'sigma_min':float(sv[-1]),'sigma_max':float(sv[0]),'condition':float(sv[0]/sv[-1]),'sensor_count':len(op.indices_1d),'sensor_speed_mean':float(vals.square().sum(-1).sqrt().mean())}
results['mean_only_rel_l2']=float(torch.linalg.norm(mean.reshape(2,64,128).permute(1,2,0)[None,None]-field)/torch.linalg.norm(field))
# Forecasting baseline on separate future frames, diagnostic only.
history=raw[50:70].permute(0,2,3,1)[None];future=raw[70:90].permute(0,2,3,1)[None]
results['dense_persistence_forecast_rel_l2']=float(torch.linalg.norm(history[:,-1:].expand_as(future)-future)/torch.linalg.norm(future))
results['parameter_counts']={}
for p in sorted(pathlib.Path('configs/yaml_main_v1').glob('*.yaml')):
    cfg=yaml.safe_load(p.read_text());op=get_sensor_placement(cfg.get('sensor_topology','wall'),cfg.get('num_sensors',64))
    model=load_model(cfg,payload,op.indices_1d)
    results['parameter_counts'][cfg['model_name']]={'total_real_scalars':sum(p.numel() for p in model.parameters()),'trainable_named_modal_prop':sum(p.numel() for p in model.modal_prop.parameters()) if hasattr(model,'modal_prop') else None}
    del model
path=pathlib.Path('reviews/review_diagnostics_20260917.json')
path.write_text(json.dumps(results,indent=2),encoding='utf-8')
print(json.dumps(results,indent=2))
