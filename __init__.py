import os
import sys
import folder_paths
import os.path as osp
now_dir = osp.dirname(__file__)
tmp_dir = osp.join(now_dir,"temp")
aifsh_dir = osp.join(folder_paths.models_dir,"AIFSH")
sys.path.append(now_dir)
minimates_models_dir = osp.join(aifsh_dir,"MiniMates")
output_dir = folder_paths.get_output_directory()
import cv2
import torch
import shutil
import torchaudio
import tempfile
import numpy as np
from PIL import Image
from pathlib import Path

class MiniMatesNode:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "avator":("IMAGE",),
                "driving_audio":("AUDIO",),
                "if_matting":("BOOLEAN",{
                    "default": False,
                    "tooltip":"if matting the person from avator image"
                })
            },
            "optional":{
                "driving_video":("VIDEO",),
            }
        }
    
    RETURN_TYPES = ("VIDEO",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "gen_video"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_MiniMates"

    def gen_video(self,avator,driving_audio,if_matting,driving_video=None):
        os.makedirs(tmp_dir,exist_ok=True)
        os.environ['minimates_models_dir'] = minimates_models_dir
        with tempfile.NamedTemporaryFile(suffix=".png",dir=tmp_dir,delete=False) as img:
            avator_np = avator.numpy()[0] * 255
            avator_np = avator_np.astype(np.uint8)
            avator_cv2 = cv2.cvtColor(avator_np,cv2.COLOR_BGR2RGBA)
            cv2.imwrite(img.name,avator_cv2)
            avator_path = Path(img.name)
        
        if if_matting:
            from minimates.interface.matting import Matting
            matting_model = Matting(osp.join(minimates_models_dir,"modnet.onnx"))
            save_img_path = osp.join(avator_path.parent,avator_path.stem+"_rgba.png")
            matting_model.predict_image_rgba(avator_path,save_img_path)
            avator_path = Path(save_img_path)

        py = sys.executable or "python"
        if driving_video:
            template_path = osp.join(output_dir,Path(driving_video).stem+".template")
            py_path = osp.join(now_dir,"minimates/interface/generate_move_template.py")
            cmd = f"""{py} {py_path} {driving_video} {template_path}"""
            if not osp.exists(template_path):
                print(cmd)
                os.system(cmd)
        else:
            template_path = None
        
        with tempfile.NamedTemporaryFile(suffix=".wav",dir=tmp_dir,delete=False) as aud:
            waveform = driving_audio['waveform'][0]
            sample_rate = driving_audio["sample_rate"]
            torchaudio.save(aud.name,waveform,sample_rate)
            audio_path = Path(aud.name)
        py_path = osp.join(now_dir,"minimates/interface/interface_audio.py")
        output_path = osp.join(output_dir,avator_path.stem+"_"+audio_path.stem+".mp4")
        if template_path:
            cmd = f"""{py} {py_path} {avator_path} {audio_path} {output_path} {template_path}"""
        else:
            cmd = f"""{py} {py_path} {avator_path} {audio_path} {output_path}"""
        print(cmd)
        os.system(cmd)
        shutil.rmtree(tmp_dir)
        return (output_path, )
        

NODE_CLASS_MAPPINGS = {
    "MiniMatesNode": MiniMatesNode
}
