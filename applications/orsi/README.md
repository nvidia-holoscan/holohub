# Orsi Academy Sample Applications

This folder contains three sample applications  

1. In and out of body detection and anonymization of surgical video sample app
2. Non Organic Structure Segmentation and with AI enhanced visualization of AR over or pre-operative structures rendered with VTK sample app  
3. Multi AI (models from 1. and 2.) and AI enhanced visualization of AR over or pre-operative structures rendered with VTK sample app


### (1) Download video and ONNX model

1. Download [orsi.zip](https://nvidia-my.sharepoint.com/:u:/p/okutter/EZdPgSx1OVtJoqQGQsZF1WYB5gSTLyQn2c6N-lDKJqrLxQ?e=jsLka8). 
2. Create data folder in main holohub folder.
3. Unzip orsi.zip to data/orsi in main holohub folder. 

### (2) Build Dev Container with dependencies for Orsi Sample Apps

#### Holoscan Container from NGC:
```bash
 ./dev_container build --verbose --docker_file applications/orsi/docker/Dockerfile   --img holohub:orsi
```
#### Local Holoscan SDK Container
```bash
./dev_container build --verbose --docker_file applications/orsi/docker/Dockerfile --base_img  holoscan-sdk-dev:latest  --img holohub:orsi-sdk-local
```
### (3) Launch Dev Container 
#### Holoscan Container from NGC:

```bash
./dev_container launch --img holohub:orsi 
```
#### Local Holoscan SDK Container: 
```bash
./dev_container launch --img holohub:orsi-sdk-local --local_sdk_root PATH_TO_LOCAL_HOLOSCAN_SDK
```
### (4) Build sample apps

**1. orsi_in_out_body** 

```bash
./run build orsi_in_out_body  
```


**2. orsi_segmentation_ar** 

```bash
./run build orsi_segmentation_ar
```

**3. orsi_multi_ai_ar** 

```bash
./run build orsi_multi_ai_ar
```

### (5) Run sample apps

**1. orsi_in_out_body** 

```bash
./run launch orsi/orsi_in_out_body cpp
```

**2. orsi_segmentation_ar** 

```bash
./run launch orsi/orsi_segmentation_ar cpp
```

**3. orsi_multi_ai_ar** 

```bash
./run launch orsi/orsi_multi_ai_ar cpp
```
