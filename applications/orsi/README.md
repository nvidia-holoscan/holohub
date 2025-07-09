# Orsi Academy Sample Applications

## Introduction

This folder contains three sample applications, please refer to the respective application README for details on assets and use.

1. **[In and out of body detection and anonymization of surgical video sample app](./orsi_in_out_body/README.md):**

    Often times during surgery the camera is removed for cleaning, possibly capturing sensitive data. To mitigate this, this sample app deploys a binary classification model to accurately classify frames being in or outside of the patient's body. Due to the realtime processing time, it can be applied in streaming settings for surgical teaching to retain patient and medical staff privacy.

2. **[Non Organic Structure Segmentation and with AI enhanced visualization of AR over or pre-operative structures rendered with VTK sample app](./orsi_segmentation_ar/README.md):**

    3D models are used in surgery to improve patient outcomes. They provide information on patient specific anatomies that are not visible in the present surgical scene. Especially in robotic surgery these 3D models give great insights because they can be projected and aligned directly onto the endoscopic video. This augmented reality supports navigation for the surgeon in the console. The downside of the 3D model projection is that it occludes the surgical instruments, creating a possible hazardous situation for the patient. This application uses a deep learning segmentation model to identify non-organic items such as tools, clips and wires and projects them on top of the 3D model. This solves the occlusion problem and adds a sense of depth to the AR application. The paper describing the deployment of this application has been awarded an outstanding paper award at the joint 17th AE-CAI, 10th CARE and 6th OR 2.0 workshop, MICCAI 2023. (<https://doi.org/10.1049/htl2.12056>)
3. **[Multi AI (models from 1. and 2.) and AI enhanced visualization of AR over or pre-operative structures rendered with VTK sample app](./orsi_multi_ai_ar/README.md):**

    A Multi AI pipeline featuring both the segmentation and anonymization models.

The segmentation and Multi AI pipelines have successfully been used in multiple in human procedures:

- Robot-assisted partial nephrectomy
- Robot-assisted migrated endovascular stent removal
- Robot-assisted liver metastasectomy
- Robot-assisted pulmonary lobectomy

## Press release and related publications

- <https://www.orsi-online.com/news/orsi-academy-brings-real-time-artificial-intelligence-or-first-its-kind-augmented-reality-surgery>
- <https://www.deltacast.tv/news/news/2023/press-release-augmented-reality-in-robotic-surgery-first-in-human-with-nvidia-holoscan-and-deltacast-video-solutions>
- <https://amazingerasmusmc.nl/chirurgie/wereldprimeur-longkankeroperatie-met-augmented-reality/>
- <https://www.orsi-online.com/news/worlds-first-lung-cancer-surgery-orsi-innotechs-augmented-reality-technology>
- Hofman, J., De Backer, P.,Manghi, I., Simoens, J., De Groote, R., Van Den Bossche, H., D’Hondt, M., Oosterlinck, T., Lippens, J.,Van Praet, C.,    Ferraguti, F., Debbaut, C., Li, Z., Kutter, O., Mottrie, A., Decaestecker, K.: First-in-human real-time AI-assisted instrument deocclusion during augmented reality robotic surgery. Healthc. Technol. Lett. 1–7 (2023). <https://doi.org/10.1049/htl2.12056>
- De Backer, P., Van Praet, C., Simoens, J., et al.: Improving augmented reality through deep learning: Real-time instrument delineation in robotic renal surgery. Eur. Urol. 84(1), 86–91 (2023)

## Run the sample applications

### (1) Build and launch Dev Container with dependencies for Orsi Sample Apps

#### Holoscan Container from NGC

```bash
 ./holohub run-container orsi
```

#### Local Holoscan SDK Container

```bash
./holohub run-container --docker-file applications/orsi/Dockerfile --base-img holoscan-sdk-dev:latest --img holohub:orsi-sdk-local --local-sdk-root PATH_TO_LOCAL_HOLOSCAN_SDK
```

### (3) Build sample apps

Note that this build step will also download and prepare the required sample data. This uncompressed data requires a substantial amount of disk space (approximately 40 GB).

**1. orsi_in_out_body**

```bash
./holohub build orsi_in_out_body
```

**2. orsi_segmentation_ar**

```bash
./holohub build orsi_segmentation_ar
```

**3. orsi_multi_ai_ar**

```bash
./holohub build orsi_multi_ai_ar
```

### (4) Run sample apps

**1. orsi_in_out_body**

C++:

```bash
./holohub run orsi_in_out_body --language cpp
```

Python:

```bash
./holohub run orsi_in_out_body --language python
```

**2. orsi_segmentation_ar**

C++:

```bash
./holohub run orsi_segmentation_ar --language cpp
```

Python:

```bash
./holohub run orsi_segmentation_ar --language python
```

**3. orsi_multi_ai_ar**

C++:

```bash
./holohub run orsi_multi_ai_ar --language cpp
```

Python:

```bash
./holohub run orsi_multi_ai_ar --language python
```

### note

This application is patent pending:

- EP23163230.8: European patent application “Real-time instrument delineation in robotic surgery”
- US18/211,269: US patent application “Real-time instrument delineation in robotic surgery”

<center> <img src="./docs/orsi_logo.png" width="400"></center>
