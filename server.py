import json
from fastmcp import FastMCP
from PIL import Image
import numpy as np
import cv2
from io import BytesIO
import uuid
import os
import websocket  # NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import urllib.request
import urllib.parse
import requests
import matplotlib.pyplot as plt
from mask_to_box_node import mask_to_box_transform
# 初始化FastMCP服务器
mcp = FastMCP("Flux")
client_id = str(uuid.uuid4())
server_address = "your server address" #修改成自己的MCP服务器

def queue_prompt(prompt):
  p = {"prompt": prompt, "client_id": client_id}
  data = json.dumps(p).encode('utf-8')
  req = urllib.request.Request("http://{}/prompt".format(server_address), data=data)
  return json.loads(urllib.request.urlopen(req).read())


def get_image(filename, subfolder, folder_type):
  data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
  url_values = urllib.parse.urlencode(data)
  with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
    return response.read()


def get_history(prompt_id):
  with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
    return json.loads(response.read())


def get_images(ws, prompt):
  prompt_id = queue_prompt(prompt)['prompt_id']
  output_images = {}
  current_node = ""
  while True:
    out = ws.recv()
    if isinstance(out, str):
      message = json.loads(out)
      print(message)
      if message['type'] == 'executing':
        data = message['data']
        if data['prompt_id'] == prompt_id:
          if data['node'] is None:
            break  # Execution is done
          else:
            current_node = data['node']
    else:
      if current_node == 'save_image_websocket_node':
        images_output = output_images.get(current_node, [])
        images_output.append(out[8:])
        output_images[current_node] = images_output

  history = get_history(prompt_id)[prompt_id]
  for node_id in history['outputs']:
    node_output = history['outputs'][node_id]
    images_output = []
    if 'images' in node_output:
      for image in node_output['images']:
        image_data = get_image(image['filename'], image['subfolder'], image['type'])
        images_output.append(image_data)
    output_images[node_id] = images_output
  return output_images

def get_single_image(ws, prompt):
  prompt_id = queue_prompt(prompt)['prompt_id']
  output_images = {}
  current_node = ""
  while True:
    out = ws.recv()
    if isinstance(out, str):
      message = json.loads(out)
      print(message)
      if message['type'] == 'executing':
        data = message['data']
        if data['prompt_id'] == prompt_id:
          if data['node'] is None:
            break  # Execution is done
          else:
            current_node = data['node']
    else:
      if current_node == 'save_image_websocket_node':
        images_output = output_images.get(current_node, [])
        images_output.append(out[8:])
        output_images[current_node] = images_output

  history = get_history(prompt_id)[prompt_id]
  for node_id in history['outputs']:
    node_output = history['outputs'][node_id]
    images_output = []
    if 'images' in node_output:
      for image in node_output['images']:
        image_data = get_image(image['filename'], image['subfolder'], image['type'])
        return image_data

def upload_image(file_path: str) -> dict:
    url = "http://" + server_address + '/api/upload/image'
    try:
      with open(file_path, 'rb') as file:
        file_name: str = os.path.basename(file_path)
        prefix, suffix = os.path.splitext(file_name.lower())
        if suffix is None or len(suffix) == 0:
          print('ERROR: image suffix empty')
          return None
        suffix = suffix.replace('.', '')
        if suffix not in ['jpg', 'jpeg', 'webp', 'png']:
          print('ERROR: not support image type ', suffix)
          return None
        content_type = 'image/' + suffix
        files = {'image': (file_name, file, content_type)}
        response = requests.post(url, files=files)
        if response.status_code == 200:
          json_response = response.json()
          print("INFO: Server Response:",
                json_response)  # {"name": "\u4f55\u7fe0\u67cf.png", "subfolder": "", "type": "input"}
          return json_response
        else:
          print(f"ERROR: Failed to upload file. Status code: {response.status_code}")
    except Exception as e:
      print('ERROR: upload image failed with ', e)
    return None
def save_image_with_incremental_name(image: Image.Image, folder: str, ext="jpg"):
  """
  保存 image 到 folder 中，命名为递增数字（如 1.jpg, 2.jpg）
  """
  # 创建目标文件夹（如果不存在）
  os.makedirs(folder, exist_ok=True)

  # 获取所有已存在的同后缀文件名
  existing_numbers = []
  for fname in os.listdir(folder):
    if fname.endswith(f".{ext}"):
      try:
        number = int(os.path.splitext(fname)[0])
        existing_numbers.append(number)
      except ValueError:
        continue  # 忽略非数字文件名

  # 计算新的编号
  next_number = max(existing_numbers, default=0) + 1
  save_path = os.path.join(folder, f"{next_number}.{ext}")

  # 保存图片
  image.save(save_path)
  print(f"图片已保存为: {save_path}")
  return next_number

@mcp.tool
def fill(source_pth:str,target_pth:str,folder:str) -> dict:
  """Call this tool to perform background fusion on the specified source image and target image using the FLUX model on the ComfyUI server and save it to the specified local path and return the index number of the saved image.."""

  output_images = {}
  # source=upload_image(source_pth)['name']
  # step1:获取目标图像的Mask
  target=upload_image(target_pth)['name']
  prompt_mask["1"]["inputs"]["image"] = target
  image_mask= get_single_image(ws, prompt_mask)
  # step2：仿射变换
  nparr = np.frombuffer(image_mask, np.uint8)
  mask = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
  syn = mask_to_box_transform(mask,source_pth,target_pth)
  os.makedirs(folder, exist_ok=True)
  cv2.imwrite(os.path.join(folder,"tmp.png"), syn)#把当前处理的Mask保存为一个临时的tmp文件
  # step3：背景填充
  f_name= upload_image(os.path.join(folder,"tmp.png"))['name']
  prompt_fill["1"]["inputs"]["image"]= f_name
  image_fill= get_single_image(ws,prompt_fill)
  img = Image.open(BytesIO(image_fill))
  index = save_image_with_incremental_name(img,folder)
  output_images["index"] = index
  # for node_id in images:
  #   node_output = images[node_id]
  #   images_output = []
  #   for image in node_output:
  #     img = Image.open(BytesIO(image))
  #     index = save_image_with_incremental_name(img,folder)
  #     images_output.append(index)
  #   output_images[node_id] = images_output
  return output_images

prompt_mask = {
    "1": {
      "inputs": {
        "image": "target.jpg",
        "upload": "image"
      },
      "class_type": "LoadImage",
      "_meta": {
        "title": "Load Image"
      }
    },
    "3": {
      "inputs": {
        "face_mask": True,
        "background_mask": False,
        "hair_mask": True,
        "body_mask": True,
        "clothes_mask": True,
        "confidence": 0.4,
        "refine_mask": False,
        "images": [
          "1",
          0
        ]
      },
      "class_type": "APersonMaskGenerator",
      "_meta": {
        "title": "A Person Mask Generator"
      }
    },
    "36": {
      "inputs": {
        "filename_prefix": "ComfyUI",
        "images": [
          "37",
          0
        ]
      },
      "class_type": "SaveImage",
      "_meta": {
        "title": "Save Image"
      }
    },
    "37": {
      "inputs": {
        "mask": [
          "3",
          0
        ]
      },
      "class_type": "MaskToImage",
      "_meta": {
        "title": "Convert Mask to Image"
      }
    }
  }
prompt_fill={
    "1": {
      "inputs": {
        "image": "result1.png",
        "upload": "image"
      },
      "class_type": "LoadImage",
      "_meta": {
        "title": "Load Image"
      }
    },
    "2": {
      "inputs": {
        "face": True,
        "left_eyebrow": False,
        "right_eyebrow": False,
        "left_eye": False,
        "right_eye": False,
        "left_pupil": False,
        "right_pupil": False,
        "lips": False,
        "number_of_faces": 1,
        "confidence": 0.4,
        "refine_mask": True,
        "images": [
          "1",
          0
        ]
      },
      "class_type": "APersonFaceLandmarkMaskGenerator",
      "_meta": {
        "title": "A Person Face Landmark Mask Generator"
      }
    },
    "3": {
      "inputs": {
        "expand": -5,
        "tapered_corners": False,
        "mask": [
          "2",
          0
        ]
      },
      "class_type": "GrowMask",
      "_meta": {
        "title": "GrowMask"
      }
    },
    "4": {
      "inputs": {
        "expand": 130,
        "tapered_corners": False,
        "mask": [
          "2",
          0
        ]
      },
      "class_type": "GrowMask",
      "_meta": {
        "title": "GrowMask"
      }
    },
    "5": {
      "inputs": {
        "x": 0,
        "y": 0,
        "operation": "subtract",
        "destination": [
          "4",
          0
        ],
        "source": [
          "3",
          0
        ]
      },
      "class_type": "MaskComposite",
      "_meta": {
        "title": "MaskComposite"
      }
    },
    "8": {
      "inputs": {
        "clip_name1": "clip_l.safetensors",
        "clip_name2": "t5xxl_fp16.safetensors",
        "type": "flux"
      },
      "class_type": "DualCLIPLoader",
      "_meta": {
        "title": "DualCLIPLoader"
      }
    },
    "9": {
      "inputs": {
        "vae_name": "ae.safetensors"
      },
      "class_type": "VAELoader",
      "_meta": {
        "title": "Load VAE"
      }
    },
    "10": {
      "inputs": {
        "text": "An Identification card of a Chinese man.consistent illumination,uniform lighting",
        "clip": [
          "8",
          0
        ]
      },
      "class_type": "CLIPTextEncode",
      "_meta": {
        "title": "CLIP Text Encode (Prompt)"
      }
    },
    "11": {
      "inputs": {
        "text": "distorted face,unnatural eyes,asymmetrical facial features, blurry face,artifacts,mutated face, double eyes,pixelation, glitches,extra limbs,compression artifacts,double mouth,incorrect background,inconsistent skin tone, artistic style,digital painting,inconsistent card background.",
        "clip": [
          "8",
          0
        ]
      },
      "class_type": "CLIPTextEncode",
      "_meta": {
        "title": "CLIP Text Encode (Prompt)"
      }
    },
    "12": {
      "inputs": {
        "guidance": 30,
        "conditioning": [
          "10",
          0
        ]
      },
      "class_type": "FluxGuidance",
      "_meta": {
        "title": "FluxGuidance"
      }
    },
    "13": {
      "inputs": {
        "noise_mask": True,
        "positive": [
          "12",
          0
        ],
        "negative": [
          "11",
          0
        ],
        "vae": [
          "9",
          0
        ],
        "pixels": [
          "1",
          0
        ],
        "mask": [
          "5",
          0
        ]
      },
      "class_type": "InpaintModelConditioning",
      "_meta": {
        "title": "InpaintModelConditioning"
      }
    },
    "14": {
      "inputs": {
        "seed": 937218906,
        "steps": 20,
        "cfg": 1,
        "sampler_name": "euler",
        "scheduler": "normal",
        "denoise": 1,
        "model": [
          "16",
          0
        ],
        "positive": [
          "13",
          0
        ],
        "negative": [
          "13",
          1
        ],
        "latent_image": [
          "13",
          2
        ]
      },
      "class_type": "KSampler",
      "_meta": {
        "title": "KSampler"
      }
    },
    "15": {
      "inputs": {
        "unet_name": "flux1-fill-dev.safetensors",
        "weight_dtype": "default"
      },
      "class_type": "UNETLoader",
      "_meta": {
        "title": "Load Diffusion Model"
      }
    },
    "16": {
      "inputs": {
        "model": [
          "15",
          0
        ]
      },
      "class_type": "DifferentialDiffusion",
      "_meta": {
        "title": "Differential Diffusion"
      }
    },
    "17": {
      "inputs": {
        "samples": [
          "14",
          0
        ],
        "vae": [
          "9",
          0
        ]
      },
      "class_type": "VAEDecode",
      "_meta": {
        "title": "VAE Decode"
      }
    },
    "19": {
      "inputs": {
        "filename_prefix": "ComfyUI",
        "images": [
          "17",
          0
        ]
      },
      "class_type": "SaveImage",
      "_meta": {
        "title": "Save Image"
      }
    }
  }

if __name__ == "__main__":
    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
    mcp.run(transport="streamable-http", host="127.0.0.1", port=22222, path="/mcp")  # Default: uses STDIO transport
    ws.close()