# Orsi Crash Reproducer 

### (1) Download video and ONNX model

1. Download [orsi.zip](https://nvidia-my.sharepoint.com/:u:/p/okutter/EV47sJ2LWxRIsJEPPKmbnVcBnaNJpSYU4L9mBv2gF3USaA?e=wxMgKa). 
2. Create data folder in main holohub folder.
3. Unzip orsi.zip to data/orsi in main holohub folder. 

### (2) Build Dev Container with dependencies 

#### Holoscan Container from NGC:
```bash
 ./dev_container build --verbose --docker_file applications/orsi_anom_only/docker/Dockerfile   --img holohub:orsi-reproducer
```

#### Local Holoscan SDK Container
```bash
./dev_container build --verbose --docker_file applications/orsi_anom_only/docker/Dockerfile --base_img holoscan-sdk-de
v:latest  --img holohub:orsi-sdk-local-reproducer
```
### (3) Launch Dev Container 

#### Holoscan Container from NGC:

```bash
./dev_container launch --img holohub:orsi-reproducer 
```
#### Local Holoscan SDK Container: 
```bash
./dev_container launch --img holohub:orsi-sdk-local-reproducer --local_sdk_root PATH_TO_LOCAL_HOLOSCAN_SDK
```

### (4) Build reproducer app 

```bash
./run build orsi_anom_only
```

### (5) Run reproducer app 

```bash
./run launch orsi_anom_only cpp 
```
See below for examplary output when running the sample app.
