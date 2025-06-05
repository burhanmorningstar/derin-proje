import { Alert } from "react-native";

// Error types
export const ERROR_TYPES = {
  NETWORK: "NETWORK_ERROR",
  PERMISSION: "PERMISSION_ERROR",
  VALIDATION: "VALIDATION_ERROR",
  API: "API_ERROR",
  UNKNOWN: "UNKNOWN_ERROR",
};

// Custom error class
export class AppError extends Error {
  constructor(message, type = ERROR_TYPES.UNKNOWN, originalError = null) {
    super(message);
    this.name = "AppError";
    this.type = type;
    this.originalError = originalError;
    this.timestamp = new Date().toISOString();
  }
}

// Error handler utility
export const handleError = (error, showAlert = true) => {
  console.error("[Error Handler]", {
    message: error.message,
    type: error.type || ERROR_TYPES.UNKNOWN,
    timestamp: error.timestamp || new Date().toISOString(),
    stack: error.stack,
    originalError: error.originalError,
  });

  if (showAlert) {
    const userMessage = getUserFriendlyMessage(error);
    Alert.alert("Hata", userMessage);
  }

  // Here you could send error to crash analytics service
  // Example: Crashlytics.recordError(error);
};

// Convert technical errors to user-friendly messages
const getUserFriendlyMessage = (error) => {
  switch (error.type) {
    case ERROR_TYPES.NETWORK:
      return "İnternet bağlantınızı kontrol edin ve tekrar deneyin.";
    case ERROR_TYPES.PERMISSION:
      return "Bu işlem için gerekli izinler verilmedi.";
    case ERROR_TYPES.API:
      return "Sunucu ile bağlantı kurulamadı. Lütfen daha sonra tekrar deneyin.";
    case ERROR_TYPES.VALIDATION:
      return "Girilen bilgiler geçerli değil.";
    default:
      return error.message || "Beklenmeyen bir hata oluştu.";
  }
};

// Network error checker
export const isNetworkError = (error) => {
  const networkMessages = [
    "network request failed",
    "fetch is not defined",
    "network error",
    "timeout",
    "connection refused",
  ];

  const errorMessage = error.message?.toLowerCase() || "";
  return networkMessages.some((msg) => errorMessage.includes(msg));
};

// API response validator
export const validateApiResponse = (response) => {
  if (!response) {
    throw new AppError("Sunucudan geçersiz yanıt alındı", ERROR_TYPES.API);
  }

  if (response.error) {
    throw new AppError(response.error, ERROR_TYPES.API);
  }

  return response;
};
