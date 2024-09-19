import os,sys
now_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(now_dir)

import shutil
import math
import folder_paths
import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download
from .infer import ViewCrafter
from viewcrafter.configs.infer_config import get_parser

output_dir = folder_paths.get_output_directory()
ckpt_dir = os.path.join(now_dir,"checkpoints")
default_traj_txt = "0 -40\n0 0\n0. -0.2"



class ViewCrafterTxTNode:
    def __init__(self):
        self.pvd = None
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "img":("IMAGE",),
                "traj_txt":(["left","loop1","loop2","wave1","zoomin1"],),
                "video_length":([25,16],),
                "ddim_steps":("INT",{
                    "default":50,
                }),
                "center_scale":("FLOAT",{
                    "default":1.,
                    "min":0.0,
                    "max":2.0,
                    "display":"slider",
                }),
                "elevation":("FLOAT",{
                    "default":5.,
                }),
                "d_theta":("INT",{
                    "default":-30,
                    "min":-40,
                    "max":40,
                    "display":"slider",
                }),
                "d_phi":("INT",{
                    "default":45,
                    "min":-45,
                    "max":45,
                    "step":5,
                    "display":"slider",
                }),
                "d_r":("FLOAT",{
                    "default":-0.5,
                    "min":-0.5,
                    "max":0.5,
                    "display":"slider",
                }),
                "seed":("INT",{
                    "default":42,
                }),
            },
            "optional":{
                "custom_traj_txt":("STRING",{
                    "multiline": True,
                    "default":default_traj_txt
                }),
            }
        }
    
    RETURN_TYPES = ("VIDEO","VIDEO",)
    RETURN_NAMES = ("result_video","traj_video",)

    FUNCTION = "gen_video"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_ViewCrafter"

    def gen_video(self,img,traj_txt,video_length,ddim_steps,center_scale,elevation,
                  d_theta,d_phi,d_r,seed,custom_traj_txt=default_traj_txt):
        parser = get_parser()
        opts = parser.parse_args()
        opts.exp_name = "AIFSH"
        opts.save_dir = os.path.join(output_dir,"viewcrafter")
        os.makedirs(opts.save_dir,exist_ok=True)
        
        img = img.numpy()[0] * 255
        img_np = img.astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        '''
        org_h, org_w = img_pil.size
        height,width = (1024,math.ceil(1024 * org_w/org_h/64)*64) if org_h > org_w else (math.ceil(1024 * org_h/org_w/64)*64,1024)
        #img_pil = img_pil.resize((height,width))
        print(f"from {(org_h,org_w)} to {(height, width)}")
        '''
        opts.height = 576
        opts.width = 1024
        tmp_img_path = os.path.join(opts.save_dir,"tmp.png")
        img_pil.save(tmp_img_path)

        opts.image_dir = tmp_img_path
        if custom_traj_txt != default_traj_txt:
            traj_path = os.path.join(opts.save_dir,"tmp.txt")
            with open(traj_path,'w',encoding="utf-8") as w:
                w.write(custom_traj_txt)
        else:
            traj_path = os.path.join(now_dir,"viewcrafter","trajs",f"{traj_txt}.txt")
        opts.traj_txt = traj_path
        opts.mode = "single_view_txt"
        opts.center_scale = center_scale
        opts.elevation = elevation
        opts.seed = seed
        opts.d_theta = d_theta
        opts.d_phi = d_phi
        opts.d_r = d_r
        ckpt_name = f"ViewCrafter_{video_length}"
        ckpt_path = os.path.join(ckpt_dir,ckpt_name,"model.ckpt")
        if not os.path.exists(ckpt_path):
            snapshot_download(repo_id=f"Drexubery/{ckpt_name}",local_dir=os.path.join(ckpt_dir,ckpt_name))
        opts.ckpt_path = ckpt_path
        opts.config = os.path.join(now_dir,"viewcrafter","configs","inference_pvd_1024.yaml")
        opts.ddim_steps = ddim_steps
        opts.video_length = video_length
        opts.device = "cuda"
        opts.model_path = os.path.join(ckpt_dir,"DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")
        if not os.path.exists(opts.model_path):
            os.system(f"wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -P {ckpt_dir}")
        
        print(opts)
        if self.pvd is None:
            self.pvd = ViewCrafter(opts)
        self.pvd.nvs_single_view()

        res_video = os.path.join(output_dir, f'{traj_txt}_diffusion0.mp4')
        shutil.copy(os.path.join(opts.save_dir, 'diffusion0.mp4'),res_video)
        traj_video = os.path.join(output_dir,f'{traj_txt}_viz_traj.mp4')
        shutil.copy(os.path.join(opts.save_dir,'viz_traj.mp4'),traj_video)
        return (res_video, traj_video,)


WEB_DIRECTORY = "./js"
from .util_nodes import PreViewVideo,LoadVideo

NODE_CLASS_MAPPINGS = {
    "LoadVideo":LoadVideo,
    "PreViewVideo":PreViewVideo,
    "ViewCrafterTxTNode": ViewCrafterTxTNode
}