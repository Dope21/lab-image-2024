# Lab 5.2 Report

## 1. Data Preparation

#### 1.1 Initial CustomImageDataset
ตั้ง `self.center_crop` เป็น `transforms.Compose` เพื่อให้แปลงเป็น tensor ก่อน และตามด้วย `transforms.CenterCrop` ตามขนาดที่ต้องการ
```python
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, gauss_noise=False, gauss_blur=False, resize=128,p=0.5):
        self.p = p
        self.resize = resize
        self.gauss_noise = gauss_noise
        self.gauss_blur = gauss_blur
        self.center_crop = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(self.resize)])
        self.image_paths = image_paths
```

#### 1.2 Implement `getitem` method
##### มีการทำงานดังนี้
- โหลดรูปตามด้วย cv2
- resize รูป โดยให้ด้านที่สั้นมีนาดยาวขั้นต่ำ 128px
- copy รูป ground truth ก่อนนำไป augment
- สุ่มความเป็นไปได้ในการใส่ noise และ blur
- normalize รูปภาพให้อยู่ใน range [0,1]
- center crop รูปภาพ
- return รูปภาพกลับไปด้วย type float32

```python

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]
        ratio = self.resize / min(h, w)
        target_h, target_w = int(h * ratio), int(w * ratio)
        image = cv2.resize(image, (target_w, target_h))

        gt_image = image.copy()

        if self.gauss_blur and np.random.rand() < self.p:
            image = self.apply_gauss_blur(image)

        if self.gauss_noise and np.random.rand() < self.p:
            image = self.apply_gauss_noise(image)
        
        gt_image = self.normalize(gt_image)
        image = self.normalize(image)

        gt_image = self.center_crop(gt_image)
        image = self.center_crop(image)

        return image.to(torch.float32), gt_image.to(torch.float32)
```
##### Gaussian Blur and Gaussian Noise
```python
    def apply_gauss_blur(self, np_image):
        kernel_size = np.random.choice([3, 5, 7, 9, 11]) # random kernel [3, 11]
        return cv2.GaussianBlur(np_image, (kernel_size, kernel_size), 0)
    
    def apply_gauss_noise(self, np_image):
        guass_mean = np.random.uniform(-50, 50) # random mean [-50, 50]
        noise = np.random.normal(guass_mean, 25.0, np_image.shape)
        return np_image + noise
```
##### Min-Max Normalize
```python
    def normalize(self, np_image):
        min = np.min(np_image, keepdims=True)
        max = np.max(np_image, keepdims=True)
        return (np_image - min) / (max - min)
```
#### 1.3 Load Dataset to Dataloader
```python
data_dir = 'D:/projects/image_processing-2024/Lab5_CNN/data/img_align_celeba'
image_paths = os.listdir(data_dir)
image_paths = [os.path.join(data_dir, img) for img in image_paths]
dataset = CustomImageDataset(
  image_paths=image_paths,
  gauss_blur=True,
  gauss_noise=True
)
dataloader = DataLoader(dataset, batch_size=16)
```
Display Image
```python
batch, gt_img = next(iter(dataloader)) 
imshow_grid(batch.numpy())
imshow_grid(gt_img.numpy())
```
Pre-process images

![preprocess images](/Lab5_CNN/pre-process-image.png)

Ground Truth Image

![ground truth images](/Lab5_CNN/gt-image.png)

## 2. Create Autoencoder
ออกแบบให้มี `DownSamplingBlock` ทั้งหมด 4 block จำนวนโหนดเริ่มตั่งแต่ 64 - 1024 และ `UpSamplingBlock` 4 block จำนวนโหนด 1024 - 64
```python
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.down1 = DownSamplingBlock(64, 128, kernel_size=3, stride=1, padding=1)
        self.down2 = DownSamplingBlock(128, 256, kernel_size=3, stride=1, padding=1)
        self.down3 = DownSamplingBlock(256, 512, kernel_size=3, stride=1, padding=1)
        self.down4 = DownSamplingBlock(512, 1024, kernel_size=3, stride=1, padding=1)

        self.up1 = UpSamplingBlock(1024, 512, kernel_size=3, stride=1, padding=1)
        self.up2 = UpSamplingBlock(512, 256, kernel_size=3, stride=1, padding=1)
        self.up3 = UpSamplingBlock(256, 128, kernel_size=3, stride=1, padding=1)
        self.up4 = UpSamplingBlock(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.conv(x)
        return x
```
## 3. Train Autoencoder
#### 3.1 Training Loop
- สร้าง training loop ตาม document ของ pytorch
- สร้างตัวแปล `avg_train_loss` สำหรับคำนวณค่า loss ของ epoch นั้น ๆ
- นำค่า loss ที่ได้ ไปบวกรวมกันไว้ในตัวแปล และหารด้วยจำนวนของ data ที่ใช้ train ทั้งหมด หลังจากจบ epoch แล้ว
```python
    for epoch in range(epochs):
        model.train()
        avg_train_loss = 0
        train_bar = tqdm(train_loader,desc=f'🚀Training Epoch [{epoch+1}/{epochs}]',unit='batch')
        for images, gt in train_bar:
            images, gt = images.to(device), gt.to(device)
            opt.zero_grad()
            outputs = model(images)
            train_loss = loss_fn(outputs, gt)
            train_loss.backward()
            opt.step()
            
            avg_train_loss += train_loss.item()
            train_bar.set_postfix(train_loss=train_loss.item())

        avg_train_loss /= len(train_loader)

        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
```
#### 3.2 Test Loop
- define `torch.no_grad()` เพื่อไม่ให้คำนวณ gradient
- หลังจากได้ outputs การ test ออกมาแล้ว จะนำไปคำนวณ `psnr` และ `ssim`
```python
        model.eval()
        with torch.no_grad():
            total_test_loss = 0
            total_psnr = 0
            total_ssim = 0
            test_bar = tqdm(test_loader,desc='📄Testing',unit='batch')
            for images, gt in test_bar:
                images, gt = images.to(device), gt.to(device)
                outputs = model(images)
                test_loss = loss_fn(outputs, gt)
                total_test_loss += test_loss.item()

                images_np = images.cpu().numpy()
                outputs_np = outputs.cpu().numpy()
                gt_np = gt.cpu().numpy()
```
- loop รูปภาพ ทั้ง `outputs` และ `gt` ออกมาทุก ๆ รูป แปลงเป็น numpy channel first และนำไปคำนวณ `psnr` และ `ssim`
- เนื่องจากใน 1 bacth มีหลายรูปภาพ จึงนำค่าที่คำนวณได้ไปบวกใส่ตัวแปล `batch_psnr` และ `batch_ssim` และหา mean โดยการหารด้วย batch size
```python
                batch_psnr = 0
                batch_ssim = 0
                for i in range(images_np.shape[0]):
                    gt_np_tranpose = gt_np[i].transpose(1, 2, 0)
                    outputs_np_transpose = outputs_np[i].transpose(1, 2, 0)
                    batch_psnr += psnr(gt_np_tranpose[i], outputs_np_transpose[i])
                    batch_ssim += ssim(gt_np_tranpose[i], outputs_np_transpose[i], channel_axis=-1, data_range=1)
                
                total_psnr += batch_psnr / images_np.shape[0]
                total_ssim += batch_ssim / images_np.shape[0]

                test_bar.set_postfix(test_loss=test_loss.item(), psnr=total_psnr, ssim=total_ssim)
```
- คำนวณค่า `avg_test_loss`, `avg_psnr`, `avg_ssim` หลังจากจบ epoch
```python
        avg_test_loss = total_test_loss / len(test_loader)
        avg_psnr = total_psnr / len(test_loader)
        avg_ssim = total_ssim / len(test_loader)

        writer.add_scalar('Loss/Test', avg_test_loss, epoch)
        writer.add_scalar('PSNR/Test', avg_psnr, epoch)
        writer.add_scalar('SSIM/Test', avg_ssim, epoch)
```
#### 3.3 Save Checkpoint
- save checkpoint หลังจากจบ การ train epoch ที่ 10
```python
    if checkpoint_path is not None:
        os.makedirs(checkpoint_path, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_path, 'model_epoch_10.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")
```
#### 3.4 Split the dataset into training and testing sets
- แปลง test set ออกเป็น 30% และ train set 70%
- batch size เป็น 16
```python
train_files, test_files = train_test_split(files, test_size=0.3, shuffle=True, random_state=2024)

train_dataset = CustomImageDataset(image_paths=train_files, gauss_blur=True, gauss_noise=True)
test_dataset = CustomImageDataset(image_paths=test_files, gauss_blur=True, gauss_noise=True)
trainloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=16)
```
#### 3.5 Train Model
```python
model = Autoencoder()
opt = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
train(
  model,
  opt,
  loss_fn,
  trainloader,
  testloader,
  epochs=10,
  checkpoint_path='D:/projects/image_processing-2024/Lab5_CNN/checkpoint',
  device='cuda' if torch.cuda.is_available() else 'cpu'
)
```
- Ex:
```
🤖Training on cuda
🚀Training Epoch [1/10]: 100%|██████████| 1313/1313 [04:37<00:00,  4.73batch/s, train_loss=0.0115] 
📄Testing: 100%|██████████| 563/563 [01:12<00:00,  7.73batch/s, psnr=1.22e+4, ssim=353, test_loss=0.0107]  
Summary:
Train avg_loss: 0.028482862061594654
Test avg_loss: 0.011698729102210296
PSNR: 21.589528875825074
SSIM: 0.626676509696097

....
.
.
.

🚀Training Epoch [10/10]: 100%|██████████| 1313/1313 [04:40<00:00,  4.68batch/s, train_loss=0.006]  
📄Testing: 100%|██████████| 563/563 [01:18<00:00,  7.13batch/s, psnr=1.37e+4, ssim=391, test_loss=0.00651] 
Summary:
Train avg_loss: 0.006657075101789158
Test avg_loss: 0.006884410342972454
PSNR: 24.267808693655805
SSIM: 0.6939349101437008
```
#### 3.6 Load And Use The Model
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('D:/projects/image_processing-2024/Lab5_CNN/checkpoint/model_epoch_10.pth', map_location=device))
model.to(device)
model.eval()
with torch.no_grad():
  batch, gt = next(iter(testloader))
  batch_to_predict = batch.to(device)
  outputs = model(batch_to_predict)
  outputs = outputs.cpu().numpy()
  imshow_grid(batch.numpy())
  imshow_grid(gt.numpy())
  imshow_grid(outputs)
```
Pre-process

![pre-process-image-2](/Lab5_CNN/pre-process-image-2.png)

Model Output

![new](/Lab5_CNN/new.png)

## 4. Explore Feature Map
- ใช้ `FeatureExtractor` class ที่สร้างไว้ให้
- define ชื่อ layer ที่ต้องการดูใน `target_layers`
- ใช้ testdata ใส่ใน `FeatureExtractor`
```python
target_layers = ['down1.conv', 'down1.relu', 'down1.pool']
feature_extractor = FeatureExtractor(model, target_layers)

with torch.no_grad():
    batch, gt = next(iter(testloader))
    feature_maps = feature_extractor(batch.to(device))

for i, x in enumerate(feature_maps):
    visualize_feature_map(x, f"feature_{i}.png")
```
- plot ออกมา และ save เป็นรูปภาพเก็บไว้
```python
import math
def visualize_feature_map(x,base_filename):
  os.makedirs("D:/projects/image_processing-2024/Lab5_CNN/feature_maps", exist_ok=True)
  plot_dim = math.sqrt(x.shape[1])
  plot_dim = math.ceil(plot_dim)
  fig, axs = plt.subplots(plot_dim, plot_dim, figsize=(plot_dim * 2, plot_dim * 2))
  axs = axs.flatten()
  for idx, image in enumerate(x[0]):
    image_np = image.cpu().detach().numpy()
    axs[idx].imshow(image_np, cmap="gray")
    axs[idx].set_title(f"ch {idx}")
    axs[idx].axis("off")
  plt.savefig(os.path.join("D:/projects/image_processing-2024/Lab5_CNN/feature_maps", base_filename))
  plt.close()
```
#### down1.convu

![down1.convu](/Lab5_CNN/feature_maps/feature_0.png)

#### down1.relu

![down1.relu](/Lab5_CNN/feature_maps/feature_1.png)

#### down1.pool

![down1.pool](/Lab5_CNN/feature_maps/feature_2.png)
