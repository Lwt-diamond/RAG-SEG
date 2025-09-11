import gradio as gr
import torch
import numpy as np
from PIL import Image
import cv2
import faiss
import os

from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import Dinov2Model, AutoImageProcessor


device = "cuda" if torch.cuda.is_available() else "cpu"
image_size = 784
# ==== 1. 初始化模型 ====
DINOV2_PATH = "facebook/dinov2-small"
processor = AutoImageProcessor.from_pretrained(DINOV2_PATH)
model = Dinov2Model.from_pretrained(DINOV2_PATH).to(device).eval()
processor.size = {
    "height": image_size,
    "width": image_size,
}
processor.do_center_crop = False  # 取消中心裁剪
processor.do_resize = True


predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-large")

# FAISS index
current_dir = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(current_dir, "sod_cod.index")
SCORES_PATH = os.path.join(current_dir, "sod_cod_score.index.npz")

index = faiss.read_index(INDEX_PATH)
scores = np.load(SCORES_PATH)["scores"].astype("float32")
d_model = 384  # dinov2-small


def faiss_precise_search(tokens, topk=1):
    dists, idxs = index.search(tokens, topk)
    results = []
    for dist_row, idx_row in zip(dists, idxs):
        for dist, idx in zip(dist_row, idx_row):
            if dist >= 0:
                results.append((scores[idx], dist))
                break
    return np.array(results)[:, 0]


import time


def faiss_precise_search(tokens, topk=1):
    # normalize_L2(search_tokens)
    st = time.time()
    distance, idx = index.search(tokens, 1)
    # print("precise search:", time.time() - st)
    print(f"Precise search time: {time.time() - st:.4f} seconds")
    combined_results = []
    for p in range(len(distance)):
        results = [[scores[i], s] for i, s in zip(idx[p], distance[p]) if s >= 0][0]
        combined_results.append(results)
    return combined_results


# ==== 2. 推理函数 ====
def run_rag_seg(pil_img: Image.Image, seed: int = 42):
    # set seed for np, torch, random
    np.random.seed(seed)
    torch.manual_seed(seed)
    import random

    random.seed(seed)

    img = pil_img.convert("RGB")
    original_size = img.size

    inputs = processor(images=img, return_tensors="pt").to(device)
    print(f"inputs pixel_values {inputs['pixel_values'].shape}")
    with torch.no_grad():
        out = model(**inputs)

    token_vec = out.last_hidden_state[:, 1:, :].squeeze().cpu().numpy()
    print(f"token_vec shape: {token_vec.shape}")
    flat_tokens = token_vec.reshape(-1, d_model).astype("float32")
    flat_tokens = np.array(flat_tokens)
    mask_vals = faiss_precise_search(flat_tokens, topk=1)

    h = w = int(np.sqrt(flat_tokens.shape[0]))
    results = mask_vals[0 : h * w]
    mask = [item[0] for item in results]
    mask = np.array(mask)
    mask_vals = mask.reshape((w, h))
    init_mask = mask_vals.astype(np.float32)
    # save initial mask for debugging
    cv2.imwrite("init_mask.png", (init_mask * 255).astype(np.uint8))
    init_mask = cv2.resize(init_mask, (1024, 1024))
    init_mask = np.array(init_mask)
    # === 3. 根据 mask 采样正负点 ===
    pos_indices = np.argwhere(init_mask > 0.99)
    neg_indices = np.argwhere(init_mask < 0.05)
    pos_points = pos_indices[:, ::-1]  # (x,y)
    neg_points = neg_indices[:, ::-1]

    print(f"前景点数量: {len(pos_points)}, 背景点数量: {len(neg_points)}")

    # 随机采样一部分点，避免过多
    num_pos_samples = min(len(pos_points), 10)
    num_neg_samples = min(len(neg_points), 10)
    if num_pos_samples > 0:
        pos_points = pos_points[
            np.random.choice(len(pos_points), num_pos_samples, replace=False)
        ]
    if num_neg_samples > 0:
        neg_points = neg_points[
            np.random.choice(len(neg_points), num_neg_samples, replace=False)
        ]

    pos_labels = np.ones(len(pos_points))
    neg_labels = np.zeros(len(neg_points))
    input_points = np.concatenate([pos_points, neg_points], axis=0)
    input_labels = np.concatenate([pos_labels, neg_labels], axis=0)
    # 调整为SAM2要求的输入格式

    print(f"input_points  {input_points}")
    print(f"input_labels {input_labels}")
    # === 4. SAM 推理 ===
    predictor.set_image(np.array(img.resize((1024, 1024))))
    init_mask_resized = (init_mask > 0.3).astype(np.float32)
    init_mask_resized = cv2.resize(init_mask_resized, (256, 256))
    init_mask_resized = init_mask_resized[None, :, :]  # (,1,H,W)
    print(f"init_mask_resized {init_mask_resized.shape}")
    print(f"input_points {input_points.shape}")
    print(f"input_labels {input_labels.shape}")
    print("init_mask_resized", init_mask_resized.shape)
    masks, _, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        mask_input=init_mask_resized,
        multimask_output=False,
    )

    seg_mask = (masks[0] * 255).astype("uint8")
    seg_mask = cv2.resize(seg_mask, original_size)

    return Image.fromarray(np.array(seg_mask))


# ==== 3. Gradio 界面 ====
# demo = gr.Interface(
#     fn=run_rag_seg,
#     inputs=gr.Image(type="pil"),
#     outputs=gr.Image(type="pil"),
#     title="RAG-SEG: Training-Free Camouflaged Object Detection",
#     description="Upload an image, RAG-SEG retrieves and segments camouflaged objects.",
# )

# if __name__ == "__main__":
#     try:
#         demo.launch(debug=True, show_error=True)
#     except Exception as e:
#         import traceback

#         traceback.print_exc()

# demo = gr.Interface(
#     fn=run_rag_seg,
#     inputs=[
#         gr.Image(type="pil", label="Input Image"),
#         gr.Slider(0, 10000, value=42, step=1, label="Seed"),
#         # gr.Number(value=42, label="Seed", precision=0),  # 也可以改成 Slider
#     ],
#     outputs=gr.Image(type="pil", label="Segmentation Result"),
#     title="RAG-SEG: Training-Free Camouflaged Object Detection",
#     description="Upload an image, RAG-SEG retrieves and segments camouflaged objects.",
#     examples=examples,
# )

# if __name__ == "__main__":
#     try:
#         demo.launch(debug=True, show_error=True)
#     except Exception as e:
#         import traceback


#         traceback.print_exc()
# ==== 3. 自定义 Example 点击事件 ====


# ==== 2. 加载 examples ====
import random


# ==== 1. 加载示例图片路径 ====
example_dir = "images"
examples = [
    os.path.join(example_dir, f)
    for f in os.listdir(example_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]


# ==== 2. 选择图库里的图片 ====
def select_example(evt: gr.SelectData):
    # img_path 就是你点的图片路径
    idx = evt.index  # 点中的图片索引
    img_path = examples[idx]
    return Image.open(img_path), 1024  # (Input Image, 默认 seed)


# ==== 3. 界面 ====
with gr.Blocks() as demo:
    gr.Markdown("## RAG-SEG: Training-Free Camouflaged Object Detection")

    with gr.Row():
        with gr.Column():
            inp_img = gr.Image(type="pil", label="Input Image")
            seed = gr.Number(value=42, label="Seed", precision=0)
            run_btn = gr.Button("Run")
        with gr.Column():
            out_img = gr.Image(type="pil", label="Segmentation Result")

    # ==== 示例图库 ====
    gallery = gr.Gallery(
        value=examples,  # 图片路径列表
        label="Example Images",
        show_label=True,
        elem_id="gallery",
        columns=4,  # 一行 4 列
        height="auto",
        type="filepath",  # 关键：让 select 传回图片路径
        interactive=True,
    )

    # 点击图库 → 自动把图片 + seed 传到输入框
    gallery.select(fn=select_example, outputs=[inp_img, seed])

    # 点击 Run → 执行推理
    run_btn.click(fn=run_rag_seg, inputs=[inp_img, seed], outputs=out_img)

if __name__ == "__main__":
    demo.launch(debug=True, show_error=True)
