import numpy as np
import cv2
import matplotlib.pyplot as plt

from diffusion import ColorDiffusion
from config import *
from dataloader import img_paths, ColorDataset


# Load model
colordiff_model = ColorDiffusion(unet_config, beta_schedule)
colordiff_model.set_new_noise_schedule(device)
load_state = torch.load('./best_model.pth')
colordiff_model.load_state_dict(load_state, strict=True)
colordiff_model.eval().to(device)


# Load original image
showed_img_idx = 55
img_path = img_paths[showed_img_idx]
img_bgr = cv2.imread(img_path)

img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
img_l = img_lab[:,:,:1]
plt.imshow(img_l, cmap='gray')
plt.axis(False)
plt.savefig("e1_gray.png")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.axis(False)
plt.savefig("e2_full_color.png")


# Infering
test_imgpath = img_paths[showed_img_idx]
test_dataset = ColorDataset([test_imgpath])
test_sample = next(iter(test_dataset))

def inference(model, test_sample):
    with torch.no_grad():
        output, visuals = model.restoration(
            test_sample[1].unsqueeze(0).to(device)
        )
    return output, visuals

output, visuals = inference(colordiff_model, test_sample)
print(output.shape)
print(visuals.shape)


# Show result single image
def show_tensor_image(img_l, img_ab, is_save=False):
    img_l = img_l.permute(1, 2, 0).numpy()
    img_ab = img_ab.permute(1, 2, 0).numpy()
    img_ab = cv2.resize(img_ab, (img_l.shape[1], img_l.shape[0]), interpolation=cv2.INTER_LINEAR)
    arr_lab = np.concatenate([img_l, img_ab], axis=2)
    arr_lab = (arr_lab + 1.0) * 255 / 2
    arr_lab = np.clip(arr_lab, 0, 255).astype(np.uint8)
    arr_bgr = cv2.cvtColor(arr_lab, cv2.COLOR_LAB2BGR)
    if is_save:
      cv2.imwrite("e4_prediction.jpg", arr_bgr)
    arr_bgr = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB)
    plt.imshow(arr_bgr)
    plt.axis(False)

output, visuals = inference(colordiff_model, test_sample)
show_tensor_image(test_sample[0], output[0].cpu(), is_save=True)
plt.savefig("e5_result_single_image.png")


# Show result multiple images
num_images = len(visuals) // 2  # Assuming visuals contains both L and AB images
num_columns = min(num_images, 11)  # Ensure a maximum of 11 columns
num_rows = (num_images + num_columns - 1) // num_columns  # Calculate the number of rows

plt.figure(figsize=(num_columns * 3, num_rows * 3))  # Set the figure size based on columns and rows

count = 1
for i, visual in enumerate(visuals):
    if i % 2 == 0:
        continue
    plt.subplot(1, 11, count)
    count += 1
    show_tensor_image(test_sample[1], visual.cpu(), is_save=False)

plt.savefig("e6_result_multi_images.png")