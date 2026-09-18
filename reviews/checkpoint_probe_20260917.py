import sys,json,pathlib,torch,yaml
sys.path.insert(0,'src')
from sparse_pod_sim2real.model import load_model
from sparse_pod_sim2real.data.sensor_topology import get_sensor_placement
from sparse_pod_sim2real.model.our_model.gappy_solver import DifferentiableGappySolver
torch.set_num_threads(4)
payload=torch.load('artifacts/pod_basis_64x128.pt',map_location='cpu',weights_only=False)
cfg=yaml.safe_load(pathlib.Path('configs/yaml_main_v1/00_pod_res_unet3d_sparse.yaml').read_text())
op=get_sensor_placement('wall',64)
m=load_model(cfg,payload,op.indices_1d).eval()
m.load_state_dict(torch.load('artifacts/test_run/01_pod_res_unet3d_sparse/best_sim.pt',map_location='cpu',weights_only=False))
raw=torch.load(r'D:/hj/Y-3 S-1/POD-Sim2Real/data/tensor_cache_64x128/real/10000_15.0.h5.pt',map_location='cpu',weights_only=False)
x=raw[50:70].permute(0,2,3,1)[None]; y=raw[70:90].permute(0,2,3,1)[None]
batch={'x_full':x,'sensor_values':op.extract_sensor_values(x)}
res={'scope':'Exploratory 20-frame forecast only, existing one-epoch Sim checkpoint, no Real adaptation, not a benchmark or hyperparameter selection'}
with torch.no_grad():
    for label in ['full_input_oracle','sparse_current','sparse_sign_corrected']:
        if label=='sparse_sign_corrected':
            def aligned(base=None):
                phi=m.warping.phi_sim+m.warping.phi_perp@m.warping.A
                q,r=torch.linalg.qr(phi,mode='reduced');sgn=torch.where(torch.diag(r)>=0,1.,-1.)
                return q*sgn
            m.get_adapted_basis=aligned
        pred=m(batch,mode='full' if label=='full_input_oracle' else 'sparse')
        res[label]={'rel_l2':float(torch.linalg.norm(pred-y)/torch.linalg.norm(y)),'v_rel_l2':float(torch.linalg.norm(pred[...,1]-y[...,1])/torch.linalg.norm(y[...,1]))}
# A frame-wise diagnostic across fixed predeclared windows; no tuning.
res['projection_window_diagnostics']=[]
for file in sorted(pathlib.Path(r'D:/hj/Y-3 S-1/POD-Sim2Real/data/tensor_cache_64x128/real').glob('*.pt')):
    raw=torch.load(file,map_location='cpu',weights_only=False)
    field=torch.cat([raw[s:s+20] for s in [50,1000,2000,3000]]).permute(0,2,3,1)[None]
    solver=DifferentiableGappySolver(payload['basis'],op.indices_1d,payload['mean'],reg_lambda=1e-4)
    with torch.no_grad():
        _,o=solver.project_full_field(field);_,g=solver(op.extract_sensor_values(field))
    res['projection_window_diagnostics'].append({'file':file.name,'oracle_rel_l2':float(torch.linalg.norm(o-field)/torch.linalg.norm(field)),'wall_gappy_rel_l2':float(torch.linalg.norm(g-field)/torch.linalg.norm(field))})
pathlib.Path('reviews/checkpoint_probe_20260917.json').write_text(json.dumps(res,indent=2),encoding='utf-8')
print(json.dumps(res,indent=2))
