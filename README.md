Urban Street Scene Description Assistant

AI-Powered Street Scene Understanding and Privacy-Aware Report Generation

Problem Statement ID: PS 112

📌 Project Overview

The Urban Street Scene Description Assistant is a Generative AI-powered application designed to analyze urban street photographs and generate structured, context-aware descriptions of road infrastructure, traffic conditions, pedestrian activity and surrounding environments.

The system combines pretrained YOLOP for driving scene perception with Qwen-VL for contextual understanding and natural-language generation.

To preserve privacy, Haar Cascade classifiers and OpenCV are used to detect and blur human faces and vehicle registration plates before AI analysis.

The application provides a React-based user interface and generates downloadable PDF reports using ReportLab.

🎯 Objectives

- Automate urban street scene analysis using artificial intelligence.
- Detect vehicles, lane markings and drivable road areas.
- Generate meaningful descriptions of visible street features.
- Describe pedestrian activity and traffic conditions.
- Protect sensitive personal information through image blurring.
- Generate structured JSON outputs and downloadable PDF reports.
- Reduce computational requirements by utilizing pretrained models.

 🚀 Key Features

 Image Upload and Preprocessing

- Upload street photographs through an interactive web interface.
- Support standard image formats such as JPG, JPEG and PNG.
- Perform image resizing, validation and color-space conversion.

Privacy Protection

- Human face detection using Haar Cascade.
- Vehicle number plate detection using Haar Cascade.
- Gaussian blurring of detected sensitive regions using OpenCV.
- No facial recognition or vehicle registration number extraction.

Traffic and Road Analysis

Pretrained YOLOP performs three major driving perception tasks:

- Traffic object detection.
- Lane detection.
- Drivable area segmentation.

Generative AI Scene Understanding

Qwen-VL analyzes street photographs to generate contextual descriptions covering:

- Road infrastructure.
- Vehicle distribution.
- Pedestrian activity.
- Apparent traffic density.
- Weather and lighting conditions.
- Visible roadside features.
- Overall street scene description.

Automated Report Generation

- Structured JSON output.
- Validation of generated information.
- JSON-to-PDF conversion using ReportLab.
- Downloadable street scene analysis reports.

 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| React.js | Frontend development |
| JavaScript | Frontend application logic |
| Python | Backend and AI processing |
| YOLOP | Traffic object detection, lane detection and drivable area segmentation |
| Qwen-VL | Vision-Language Model and scene description |
| Haar Cascade | Face and number plate detection |
| OpenCV | Image preprocessing and privacy blurring |
| PyTorch | Model inference |
| Hugging Face Transformers | Vision-Language Model integration |
| JSON | Structured data representation |
| ReportLab | PDF report generation |
| Google Colab | AI model development and testing |
| VS Code | Application development |
| Git and GitHub | Version control and collaboration |

System Architecture

    User Uploads Street Photograph
                 |
                 v
          React.js Frontend
                 |
                 v
           Python Backend
                 |
                 v
       OpenCV Image Preprocessing
                 |
                 v
       Haar Cascade Detection
          /              \
    Human Faces      Number Plates
          \              /
                 |
                 v
           Gaussian Blur
                 |
                 v
       Privacy-Processed Image
                 |
                 v
          Pretrained YOLOP
                 |
       +---------+----------+
       |         |          |
    Vehicle    Lane      Drivable
    Detection  Detection Area Segmentation
       |         |          |
       +---------+----------+
                 |
                 v
              Qwen-VL
       Contextual Scene Analysis
                 |
                 v
         Structured JSON Output
                 |
                 v
          Output Validation
                 |
                 v
             ReportLab
          PDF Report Generation
                 |
                 v
       React Results Dashboard

⚙️ Project Workflow

1. The user uploads an urban street photograph through the React frontend.
2. OpenCV preprocesses the uploaded image.
3. Haar Cascade classifiers detect human faces and vehicle number plates.
4. Gaussian Blur is applied to the detected sensitive regions.
5. The privacy-processed photograph is passed to pretrained YOLOP.
6. YOLOP performs traffic object detection, lane detection and drivable area segmentation.
7. Qwen-VL analyzes the processed photograph and available detection information.
8. The generated street scene description is organized into JSON format.
9. ReportLab converts the structured JSON data into a formatted PDF report.
10. The results and downloadable report are presented through the React frontend.

 🧠 AI Models

YOLOP — You Only Look Once for Panoptic Driving Perception

YOLOP is a multitask deep learning model designed for driving scene perception.

The pretrained model performs:

- Traffic object detection.
- Drivable area segmentation.
- Lane detection.

The original YOLOP model was trained on the BDD100K dataset. This project utilizes pretrained weights without requiring additional custom training.

Qwen-VL — Vision-Language Model

Qwen-VL is a multimodal artificial intelligence model capable of interpreting visual information and generating natural-language responses.

In this project, Qwen-VL is responsible for:

- Contextual scene understanding.
- Pedestrian activity description.
- Road infrastructure interpretation.
- Traffic context analysis.
- Structured description generation.

Haar Cascade Classifiers

Haar Cascade is a machine learning-based object detection approach used in the privacy preprocessing module.

Separate pretrained classifiers are used to detect:

- Human faces.
- Vehicle registration plates.

The detected regions are blurred using OpenCV before further processing.

📂 Data Preparation

The project follows a pretrained-model-based inference approach. Therefore, a large-scale custom training dataset is not required.

A representative collection of labelled sample street photographs is used or planned for functional testing and performance evaluation.

The evaluation images cover:

- Urban roads and intersections.
- Heavy and light traffic conditions.
- Daytime and nighttime environments.
- Different weather conditions.
- Pedestrian-heavy streets.
- Visible lane markings.
- Images containing human faces and vehicle registration plates.

The sample images are used for evaluation rather than model training.

📄 PDF Report Generation

ReportLab converts structured JSON output into a formatted PDF report containing:

- Street scene overview.
- Road infrastructure analysis.
- Traffic object information.
- Lane and drivable area information.
- Pedestrian activity description.
- Environmental conditions.
- Privacy processing information.
- AI-generated scene description.

 🔐 Privacy and Ethical Considerations

The application follows a privacy-by-design approach.

- Human faces and vehicle registration plates are targeted for blurring before generative AI analysis.
- Personal identities are not extracted or disclosed.
- Vehicle registration numbers are not transcribed.
- AI-generated descriptions focus on visible street features.
- Uncertain visual information should be reported as unknown rather than fabricated.
- Original photographs containing sensitive information should not be publicly exposed or committed to the repository.

Testing and Evaluation

The system is evaluated using representative sample street photographs.

Evaluation criteria include:

| Module | Evaluation Criteria |
|---|---|
| YOLOP | Traffic object detection accuracy |
| Lane detection | Lane segmentation quality |
| Drivable area segmentation | Road region segmentation quality |
| Privacy module | Face and number plate detection coverage |
| Qwen-VL | Description accuracy and contextual relevance |
| JSON generation | Structural validity and completeness |
| ReportLab | PDF formatting and report generation |
| Overall system | Inference time and functional reliability |

 Limitations

- Original YOLOP does not provide a dedicated pedestrian detection class.
- Pedestrian activity descriptions generated by Qwen-VL may require manual verification.
- Haar Cascade classifiers may miss small, occluded or differently oriented faces and number plates.
- Gaussian blurring does not guarantee complete anonymization.
- Model performance may decrease under poor lighting and challenging weather conditions.
- Generative AI models may produce inaccurate descriptions.
- A single photograph cannot reliably determine vehicle speed or real-time traffic flow.
- GPU availability may affect inference performance.

 Potential Applications

- Smart city development.
- Urban infrastructure assessment.
- Road safety analysis.
- Traffic scene documentation.
- Urban planning.
- Road condition monitoring.
- Automated street scene reporting.

🏆 Hackathon Project

Problem Statement: PS 112 – Urban Street Scene Description Assistant

This project demonstrates the integration of computer vision, Generative AI, privacy-aware image processing and automated report generation for urban street scene analysis.

---

**Built with Computer Vision, Generative AI and React.js.**
