import os
import uuid
import nibabel as nb
import numpy as np
from flask import Flask, request, jsonify, render_template
from google.cloud import storage
import io
import imageio
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared_utils.logger_config import get_logger
from dotenv import load_dotenv

load_dotenv()




EPSILON = 1e-8

# Instanciar el logger para este servicio específico
logger = get_logger("ingestion_service")



app = Flask(__name__)

# CONFIGURACIÓN DINÁMICA
# Si no encuentra la variable, por defecto es 'LOCAL'
ENV = os.getenv("ENV", "LOCAL")
BUCKET_NAME = os.getenv("BUCKET_NAME", "mi-bucket-tesis")
LOCAL_STORAGE_PATH = "local_storage" # Carpeta para pruebas en tu PC
key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if key_path and os.path.exists(key_path):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
    logger.info(f"🔐 Credenciales cargadas desde: {key_path}")
else:
    # Si no hay llave y estamos en CLOUD, asumimos que estamos dentro de GCP (Cloud Run)
    if ENV == "CLOUD":
        logger.info("☁️ Cloud Auth: Usando identidad nativa de GCP (Cloud Run/Functions)")
    else:
        logger.warning("⚠️ Sin credenciales locales y sin entorno Cloud. El proceso podría fallar.")

    

# Crear carpeta local si no existe
if ENV == "LOCAL" and not os.path.exists(LOCAL_STORAGE_PATH):
    os.makedirs(LOCAL_STORAGE_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    return "OK", 200

@app.route('/upload', methods=['POST'])
def upload_nifti():
    
    file = request.files.get('file')
    study_id = str(uuid.uuid4())
    logger.info(f"THE PROCESS FOR THE {study_id} STARTS")
    
    temp_nii = f"temp_{study_id}.nii"
    file.save(temp_nii)
    
    try:
        data = nb.load(temp_nii, mmap = False)
        data = nb.as_closest_canonical(data)
        volume = data.get_fdata()
        shape = volume.shape
        slice_axis = np.argmin(shape)
        imgs = np.moveaxis(volume,slice_axis,0)
        logger.info(f"The final shape for the imgs array is: {imgs.shape}")
        if ENV == "CLOUD":
            logger.info("PROCES ON CLOUD ENV")
            client = storage.Client()
            bucket = client.bucket(BUCKET_NAME)
            base_path = f"staging/{study_id}"
            for i, img in enumerate(imgs, start = 1):
                img_buffer = io.BytesIO()
                img_min = np.min(img)
                img_max = np.max(img)
                img_range = img_max - img_min
                logger.info(f"The shappe of the images is : {img.shape}")
                if img_range == 0:
                    img_final = np.zeros(img.shape, dtype=np.uint8)
                    logger.warning(f"Slice {i} is empty or constant. Generating black image.")
                else:
        
                    img_norm = (img - img_min) / (img_range + EPSILON)
                    img_final = (img_norm * 255).astype(np.uint8)

                img_norm = (img - np.min(img)) / (np.max(img) - np.min(img))
                img_final = (img_norm * 255).astype(np.uint8)
                imageio.imsave(img_buffer, img_final, format = "PNG")
                img_buffer.seek(0)
                slice_name = str(i).zfill(3)
                blob = bucket.blob(f"staging/{study_id}/slice_{slice_name}.png")
                blob.upload_from_string(img_buffer.getvalue(), content_type="image/png")
                
                pass
            logger.info("PROCESS COMPLETED SUCCESSS")

            return jsonify({"status": "Cloud Upload Success", 
                            "id": study_id,
                             "folder_path" : f"gs://{BUCKET_NAME}/{base_path}" })
        
        
        else:
            logger.info("PROCES ON LOCAL ENV")
            study_folder = os.path.join(LOCAL_STORAGE_PATH, study_id)
            os.makedirs(study_folder)
            
            for i, img in enumerate(imgs, start = 1):
                
                img_buffer = io.BytesIO()
                img_min = np.min(img)
                img_max = np.max(img)
                img_range = img_max - img_min
                logger.info(f"The shappe of the images is : {img.shape}")
                if img_range == 0:
                    img_final = np.zeros(img.shape, dtype=np.uint8)
                    logger.warning(f"Slice {i} is empty or constant. Generating black image.")
                else:
                    img_norm = (img - img_min) / (img_range + EPSILON)
                    img_final = (img_norm * 255).astype(np.uint8)


                slice_path = os.path.join(study_folder, f"slice_{i}.png")
                imageio.imsave(slice_path,img_final, format="PNG")
            
            logger.info("PROCESS COMPLETED SUCCESSS")
            return jsonify({
                "status": "Local Save Success",
                "id": study_id,
                "folder_path": os.path.abspath(study_folder)
            })
        

    except Exception as e : 
        logger.error(f"CRITICAL ERROR IN STUDY {study_id}, {e}",exc_info = True)
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        if os.path.exists(temp_nii):
            os.remove(temp_nii)
            logger.info(f"REMOVE THE TEMPROAL {temp_nii}")

    
    


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))

