import os
import logging
from google.cloud import logging as cloud_logging

def get_logger(service_name):
    """Configura y retorna un logger basado en el entorno."""
    logger = logging.getLogger(service_name)
    logger.setLevel(logging.INFO)
    
    # Evitar duplicidad de logs si el logger ya fue configurado
    if logger.handlers:
        return logger

    env = os.getenv("APP_ENV", "LOCAL")

    if env == "CLOUD":
        # Integración nativa con GCP
        client = cloud_logging.Client()
        client.setup_logging()
        # En CLOUD, gcloud logging ya maneja el formato, no necesitamos Formatter extra
    else:
        # Formato legible para desarrollo local en terminal
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger