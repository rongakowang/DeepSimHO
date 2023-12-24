import os
import mujoco
import torch
import torch.multiprocessing as mp
import tempfile
import pytorch3d.io
import trimesh
from copy import deepcopy
from anakin.opt import cfg
from model.rotation import *
from manotorch.manolayer import ManoLayer

os.environ["DISPLAY"] = ":1" # disable display

OBJ_ROT_DIM = 4 
MODEL_NV = 7

class MuJoCoMeshWorld:
  # inline hard code class for efficiency
  def template_xml(meshn, camera_dir="camera"):
    ap = 3.14159265359 if camera_dir == 'camera' else 0 # otherwise OpenGL frame
    gravity = 9.8 if camera_dir == 'camera' else -9.8
    light_direction = '0 1.0 4' if camera_dir == 'camera' else '0 -1.0 -4'
    ligh_pos = '0 -1.0 -4' if camera_dir == 'camera' else '0 1.0 4'

    OBJ_ASSET = """<mesh file="obj_mesh.obj"/>\n"""
    OBJ_QUAT_BODY = """<geom mesh="obj_mesh" class="visual" condim="6"/>\n"""
    for i in range(meshn):
      OBJ_ASSET += f"""        <mesh file="obj_mesh{i}.obj"/>\n"""
      OBJ_QUAT_BODY += f"""          <geom mesh="obj_mesh{i}" class="collision" condim="6"/>\n"""
    xml_string = \
    f"""<mujoco model="init">
      <compiler autolimits="true" angle="radian"/>
      <option gravity="0 {gravity} 0"/>
      <default>
        <default class="visual">
          <geom group="2" type="mesh" contype="0" conaffinity="0"/>
        </default>
        <default class="collision">
          <geom group="3" type="mesh"/>
        </default>
      </default>

      <asset>
        <texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2"
          width="512" height="512"/>	
        <material name='MatGnd' reflectance='.1' texture="texplane" texrepeat="2 2" texuniform="true"/>  
        <material name="floor" reflectance=".1"/>

        <mesh file="mesh0.obj"/>
        <mesh file="mesh1.obj"/>
        <mesh file="mesh2.obj"/>
        <mesh file="mesh3.obj"/>
        <mesh file="mesh4.obj"/>
        <mesh file="mesh5.obj"/>
        <mesh file="mesh6.obj"/>
        <mesh file="mesh7.obj"/>
        <mesh file="mesh8.obj"/>
        <mesh file="mesh9.obj"/>
        <mesh file="mesh10.obj"/>
        <mesh file="mesh11.obj"/>
        <mesh file="mesh12.obj"/>
        <mesh file="mesh13.obj"/>
        <mesh file="mesh14.obj"/>
        <mesh file="mesh15.obj"/>

        {OBJ_ASSET}
      </asset>

      
      <worldbody>
        <camera name="camera1" pos="0 0 0" euler="{ap} 0 0"/>
        <light directional='false' diffuse='.8 .8 .8' specular='0.3 0.3 0.3' pos='{ligh_pos}' dir='{light_direction}'/>

        <body name="obj">
          <freejoint />
          {OBJ_QUAT_BODY}
        </body>

        <body name="hand">
            <geom name="mesh0" mesh="mesh0" type="mesh"/>
            <geom name="mesh1" mesh="mesh1" type="mesh"/>
            <geom name="mesh2" mesh="mesh2" type="mesh"/>
            <geom name="mesh3" mesh="mesh3" type="mesh"/>
            <geom name="mesh4" mesh="mesh4" type="mesh"/>
            <geom name="mesh5" mesh="mesh5" type="mesh"/>
            <geom name="mesh6" mesh="mesh6" type="mesh"/>
            <geom name="mesh7" mesh="mesh7" type="mesh"/>
            <geom name="mesh8" mesh="mesh8" type="mesh"/>
            <geom name="mesh9" mesh="mesh9" type="mesh"/>
            <geom name="mesh10" mesh="mesh10" type="mesh"/>
            <geom name="mesh11" mesh="mesh11" type="mesh"/>
            <geom name="mesh12" mesh="mesh12" type="mesh"/>
            <geom name="mesh13" mesh="mesh13" type="mesh"/>
            <geom name="mesh14" mesh="mesh14" type="mesh"/>
            <geom name="mesh15" mesh="mesh15" type="mesh"/>
        </body>

      </worldbody>

      <actuator>
        <adhesion name="hand_act" body="hand" ctrlrange="0 10" gain="100"/>
      </actuator>

      </mujoco>
    """
    return xml_string

  def getworld(obj_name, camera_dir):
    if camera_dir == 'camera':
      xml, obj_asset = MuJoCoMeshWorld.dexycb_world[MuJoCoMeshWorld.obj_index[obj_name]]
    if camera_dir == 'OpenGL':
      xml, obj_asset = MuJoCoMeshWorld.ho3d_world[MuJoCoMeshWorld.obj_index[obj_name]]
    return xml, deepcopy(obj_asset)

  FILE_ROOT = os.getcwd().replace('main', '') + 'MuJoCo_data'
  obj_names = sorted([p for p in os.listdir(f'{FILE_ROOT}/skeletons_CoACD') if '0' in p])
  obj_index = {}
  dexycb_world = []
  ho3d_world = []
  for i, n in enumerate(obj_names):
    obj_index[n] = i

  for obj in obj_names:
    obj_folder = f'{FILE_ROOT}/skeletons_CoACD/{obj}/'
    meshes = [p for p in os.listdir(obj_folder) if p.endswith('.obj')]
    obj_asset = {}
    for m in meshes:
      with open(f'{FILE_ROOT}/skeletons_CoACD/{obj}/{m}', 'rb') as f:
        obj_asset[m] = f.read() 

    dex_xml = template_xml(len(meshes) - 1, 'camera')
    dexycb_world.append((dex_xml, obj_asset))

    ho3d_xml = template_xml(len(meshes) - 1, 'OpenGL')
    ho3d_world.append((ho3d_xml, obj_asset))



class MuJoCoMeshSimulatorFast:

  def get_mapping(mano_layer):
    finger_index = torch.argmax(mano_layer.th_weights, dim=-1)
    verts_mapping = {}
    for idx, k in enumerate(finger_index):
        kp = k.item()
        if kp in verts_mapping:
            verts_mapping[kp].add(idx)
        else:
            verts_mapping[kp] = set([idx])

    faces = mano_layer.th_faces

    face_mapping = {}

    for f in faces:
        for k, v in verts_mapping.items():
            for v_idx in f:
                v_idx_p = v_idx.item()
                if v_idx_p in v:
                    face_mapping.setdefault(k, []).append(f)
                    break

    for k, v in face_mapping.items():
        for f in v:
            for v_idx in f:
                v_idxp = v_idx.item()
                verts_mapping[k].add(v_idxp)

    for k, v in verts_mapping.items():
        verts_mapping[k] = sorted(list(v))

    face_mapping_reshape = {}

    for k,v in face_mapping.items():
        for jdx, f in enumerate(v):
            t = torch.zeros(3).long()
            for i, vdx in enumerate(f):
                vdxp = vdx.item()
                t[i] = verts_mapping[k].index(vdxp)
            face_mapping_reshape.setdefault(k, []).append(t)

    for k, v in face_mapping_reshape.items():
        face_mapping_reshape[k] = torch.stack(v, dim=0)

    return verts_mapping, face_mapping_reshape

  mano_layer = ManoLayer(
            ncomps=15,
            center_idx=9,
            side='right',
            mano_assets_root='assets/mano_v1_2',
            use_pca=True,
            flat_hand_mean=False,
        )
  verts_mapping, face_mapping_reshape = get_mapping(mano_layer)

  def init_state(obj_name, mano_verts_abs, obj_rot, obj_trans, camera_dir, with_debug=False):
    mano_verts_abs = mano_verts_abs.detach().reshape(778, 3)
    if torch.is_tensor(obj_rot):
      obj_rot = obj_rot.detach().cpu().numpy()
      obj_trans = obj_trans.detach().cpu().numpy()

    obj_rot = obj_rot.reshape(OBJ_ROT_DIM,)
    obj_trans = obj_trans.reshape(3,)
    
    xml, obj_asset = MuJoCoMeshWorld.getworld(obj_name, camera_dir) # get data
    hand_asset = {}
    ct = 0
    for k, v in MuJoCoMeshSimulatorFast.verts_mapping.items():
      v0 = mano_verts_abs[v, :]
      f0 = MuJoCoMeshSimulatorFast.face_mapping_reshape[k].int()
      with tempfile.NamedTemporaryFile(suffix='.obj') as file_obj:
            pytorch3d.io.save_obj(file_obj.name, verts=v0.clone().detach(), faces=f0.clone().detach())
            hand_asset[f'mesh{ct}.obj'] = file_obj.read()
      ct += 1

    ASSET = {}
    ASSET.update(obj_asset)
    ASSET.update(hand_asset)
    try:
      model = mujoco.MjModel.from_xml_string(xml, ASSET) # always create new model here
    except:
      print("use convex instead")
      ct = 0
      hand_asset = {}
      for k, v in MuJoCoMeshSimulatorFast.verts_mapping.items():
        v0 = mano_verts_abs[v, :].cpu().numpy()
        convex_m = trimesh.convex.convex_hull(v0, qhull_options='QbB Pp Qt')
        convex_v = torch.tensor(convex_m.vertices)
        convex_f = torch.tensor(convex_m.faces).int()
        with tempfile.NamedTemporaryFile(suffix='.obj') as file_obj:
              pytorch3d.io.save_obj(file_obj.name, verts=convex_v.clone().detach(), faces=convex_f.clone().detach())
              hand_asset[f'mesh{ct}.obj'] = file_obj.read()
        ct += 1
      
      ASSET = {}
      ASSET.update(obj_asset)
      ASSET.update(hand_asset)
      model = mujoco.MjModel.from_xml_string(xml, ASSET) # always create new model here
      
    data = mujoco.MjData(model)

    data.qpos[:MODEL_NV - OBJ_ROT_DIM] = obj_trans
    data.qpos[MODEL_NV - OBJ_ROT_DIM:] = obj_rot
    
    if with_debug:
      return model, data, hand_asset, xml, data.qpos
    else:
      return model, data

  def forward(args):
    obj_name, mano_verts_abs, obj_rot, obj_trans, camera_dir = args
    sim_step = cfg['PHYSICS']['SIM_STEP']
    model, data = MuJoCoMeshSimulatorFast.init_state(obj_name, mano_verts_abs, obj_rot, obj_trans, camera_dir)
    data.ctrl = np.array([10.]).reshape(1,) # set control force for adhesion
    mujoco.mj_step(model, data, sim_step)
    final_state = np.concatenate([data.qpos, data.qvel])
    final_result = torch.tensor(final_state) # does not requires grad
    return final_result

  def get_contact_info_scratch(args):
    obj_name, mano_verts_abs, obj_rot, obj_trans, camera_dir = args
    model, data = MuJoCoMeshSimulatorFast.init_state(obj_name, mano_verts_abs, obj_rot, obj_trans, camera_dir)
    mujoco.mj_forward(model, data) # no step
    contacts = data.contact
    ncontact = data.ncon
    if not contacts.dist.any():
      max_penetration = 0
    else:
      max_penetration = abs(min(contacts.dist)) # in cm
    return ncontact, max_penetration # return number of contact and penetration
    
  def get_contact_info_complete(args):
    obj_name, mano_verts_abs, obj_rot, obj_trans, camera_dir = args
    model, data = MuJoCoMeshSimulatorFast.init_state(obj_name, mano_verts_abs, obj_rot, obj_trans, camera_dir)
    mujoco.mj_forward(model, data) # no step
    contacts = data.contact
    if data.ncon == 0:
      return []
    else:
      contact_summary = []
      for cp in contacts:
        contact_summary.append((-cp.dist, cp.frame, cp.pos - obj_trans)) # frame is a unit vector, pos relative to the center of obj mass
    return contact_summary

  @staticmethod
  def batched_simulate(obj_name, mano_verts_abs, obj_rot, obj_trans, camera_dir):
      batch_size = mano_verts_abs.shape[0]
      pool = mp.Pool()

      args_list = [(obj_name[i], mano_verts_abs[i].detach().cpu(),
              obj_rot[i].detach().cpu().numpy(),
              obj_trans[i].detach().cpu().numpy(), camera_dir) for i in range(batch_size)]
      
      final_results = pool.map(MuJoCoMeshSimulatorFast.forward, args_list)
      pool.close()
      pool.join()

      final_results = torch.Tensor(np.array([result.numpy() for result in final_results])).cuda()
      return final_results

  @staticmethod
  def batched_get_contact(obj_name, mano_verts_abs, obj_rot, obj_trans, camera_dir): # forward function must create as a tensor
    # each is a torch tensor of size (B, -1) 
    batch_size = mano_verts_abs.shape[0]
    pool = mp.Pool()

    args_list = [(obj_name[i], mano_verts_abs[i].detach().cpu(),
              obj_rot[i].detach().cpu().numpy(),
              obj_trans[i].detach().cpu().numpy(), camera_dir) for i in range(batch_size)]
      
    final_results = pool.map(MuJoCoMeshSimulatorFast.get_contact_info_scratch, args_list)
    pool.close()
    pool.join()

    b_ncon = torch.Tensor([float(result[0]) for result in final_results]).cuda()
    b_pen = torch.Tensor([float(result[1]) for result in final_results]).cuda()
    return b_ncon, b_pen

  @staticmethod
  def batched_simulate_and_get_contact(obj_name, mano_verts_abs, obj_rot, obj_trans, camera_dir):
      batch_size = mano_verts_abs.shape[0]
      pool = mp.Pool()

      args_list = [(obj_name[i], mano_verts_abs[i].detach().cpu(),
              obj_rot[i].detach().cpu().numpy(),
              obj_trans[i].detach().cpu().numpy(), camera_dir) for i in range(batch_size)]
      
      final_results = pool.map(MuJoCoMeshSimulatorFast.forward, args_list) # just use normal forward
      contact_results = pool.map(MuJoCoMeshSimulatorFast.get_contact_info_complete, args_list)
      pool.close()
      pool.join()

      final_results = torch.Tensor([result.detach().numpy() for result in final_results]).cuda()
      contact_results = [result for result in contact_results]
      return final_results, contact_results