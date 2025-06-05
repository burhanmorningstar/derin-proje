import { CONFIG } from "../constants/config";
import {
  AppError,
  ERROR_TYPES,
  isNetworkError,
  validateApiResponse,
} from "../utils/errorHandler";

class ApiService {
  constructor() {
    this.baseURL = CONFIG.API.BASE_URL;
    this.timeout = CONFIG.API.TIMEOUT;
  }
  // Generic fetch wrapper with error handling
  async makeRequest(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;

    // Create abort controller for timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new AppError(
          `HTTP ${response.status}: ${response.statusText}`,
          ERROR_TYPES.API
        );
      }

      const data = await response.json();
      return validateApiResponse(data);
    } catch (error) {
      clearTimeout(timeoutId);

      if (error.name === "AbortError") {
        throw new AppError("İstek zaman aşımına uğradı", ERROR_TYPES.NETWORK);
      }

      if (isNetworkError(error)) {
        throw new AppError("Ağ bağlantısı hatası", ERROR_TYPES.NETWORK, error);
      }

      if (error instanceof AppError) {
        throw error;
      }

      throw new AppError("Beklenmeyen API hatası", ERROR_TYPES.API, error);
    }
  }
  // Send frame for prediction
  async sendFrameForPrediction(frameUri) {
    const formData = new FormData();
    formData.append("frames", {
      uri: frameUri,
      type: "image/jpeg",
      name: "frame.jpg",
    });

    return this.makeRequest(CONFIG.API.ENDPOINTS.PREDICT, {
      method: "POST",
      body: formData,
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });
  }

  // Health check
  async healthCheck() {
    try {
      const response = await fetch(`${this.baseURL}/status`, {
        method: "GET",
        timeout: 5000,
      });
      return response.ok;
    } catch (error) {
      return false;
    }
  }
}

export default new ApiService();
