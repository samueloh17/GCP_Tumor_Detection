import os 
from dotenv import load_dotenv
from google.cloud import aiplatform


load_dotenv()



GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")
BUCKET_MODELS = os.getenv("BUCKET_MODELS")

def register_medical_models():
    aiplatform.init(project=PROJECT_ID, location=REGION)

    print("🚀 Iniciando registro de modelos en Vertex AI Model Registry...")

    model_unet = aiplatform.Model.upload(
        display_name="unet-tumor-segmentation",
        artifact_uri=f"{BUCKET_MODELS}/unet/v1/",
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-12:latest",
        description="Modelo U-Net para segmentación del cebrebro a  craneo, ojos, nariz, etc; en imágenes NIfTI (PNG slices).",
        labels={"task": "segmentation", "architecture": "unet"}
    )
    print(f"✅ U-Net registrada con éxito. ID: {model_unet.resource_name}")



if __name__ == "__main__":
    register_medical_models()


    


