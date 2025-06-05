import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import logging
from config import MODEL_PATH, DEVICE, TRANSFORM, SEQUENCE_LENGTH
from model_utils import CNNLSTM

logger = logging.getLogger(__name__)

class ModelService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the model and transforms."""
        try:
            self.device = torch.device(DEVICE)
            self.model = CNNLSTM().to(self.device)
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            self.model.eval()

            self.transform = transforms.Compose([
                transforms.Resize(TRANSFORM["resize"]),
                transforms.ToTensor() if TRANSFORM["to_tensor"] else transforms.Lambda(lambda x: x)
            ])

            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    def predict(self, image: Image.Image) -> dict:
        """
        Make a prediction on a single image.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            dict: Prediction results with probabilities and label
        """
        try:
            # Preprocess image
            image = self.transform(image)
            
            # Create sequence of frames (repeating the same frame)
            frames = torch.stack([image for _ in range(SEQUENCE_LENGTH)])
            frames = frames.unsqueeze(0).to(self.device)

            # Make prediction
            with torch.no_grad():
                output = self.model(frames)
                probs = torch.softmax(output, dim=1)[0].cpu().numpy()
                pred = int(np.argmax(probs))
                labels = ['NonViolence', 'Violence']

            return {
                "violence_prob": float(probs[1]),
                "nonviolence_prob": float(probs[0]),
                "prediction": labels[pred]
            }
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise 