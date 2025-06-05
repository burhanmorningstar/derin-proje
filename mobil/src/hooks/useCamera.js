import { useState, useRef, useCallback, useEffect } from "react";
import { Camera, useCameraPermissions } from "expo-camera";
import { Alert } from "react-native";
import { CONFIG, MESSAGES } from "../constants/config";
import {
  ApiService,
  handleError,
  AppError,
  ERROR_TYPES,
  calculateOptimalImageSize,
  cleanupImageUri,
} from "../utils";
import { useMountedRef } from "./usePerformance";

export const useCamera = () => {
  const [permission, requestPermission] = useCameraPermissions();
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [intervalId, setIntervalId] = useState(null);
  const [predictions, setPredictions] = useState([]);

  const cameraRef = useRef(null);
  const isMounted = useMountedRef();
  const isCameraActiveRef = useRef(false); // Add ref to track camera state

  // Request camera permissions
  const requestPermissions = useCallback(async () => {
    try {
      if (!permission?.granted) {
        const { granted } = await requestPermission();
        if (!granted) {
          throw new AppError(
            "Kamera erişim izni gerekli!",
            ERROR_TYPES.PERMISSION
          );
        }
        return granted;
      }
      return true;
    } catch (error) {
      const appError =
        error instanceof AppError
          ? error
          : new AppError(
              "Kamera izni alınamadı",
              ERROR_TYPES.PERMISSION,
              error
            );
      setError(appError);
      handleError(appError);
      return false;
    }
  }, [permission, requestPermission]); // Capture and send camera frame
  const captureAndSendFrame = useCallback(async () => {
    console.log("🎯 Attempting to capture frame...");
    console.log("Camera ref exists:", !!cameraRef.current);
    console.log("Camera active (state):", isCameraActive);
    console.log("Camera active (ref):", isCameraActiveRef.current);

    if (!cameraRef.current || !isCameraActiveRef.current) {
      console.log("❌ No camera ref or camera not active");
      return;
    }

    try {
      setIsLoading(true);
      setError(null);
      console.log("📷 Taking picture...");

      // Take picture - try new expo-camera API first
      let photo;
      try {
        // New expo-camera API
        photo = await cameraRef.current.takePictureAsync({
          quality: CONFIG.VIDEO.COMPRESSION_QUALITY,
          base64: false,
          skipProcessing: false,
        });
      } catch (firstError) {
        console.log("⚠️ First method failed, trying alternative...");
        try {
          // Alternative method
          photo = await cameraRef.current.takePicture({
            quality: CONFIG.VIDEO.COMPRESSION_QUALITY,
            base64: false,
          });
        } catch (secondError) {
          console.error(
            "❌ Both picture methods failed:",
            firstError,
            secondError
          );
          throw new Error("Camera capture failed: " + firstError.message);
        }
      }

      console.log("✅ Picture taken:", photo.uri);

      if (!isMounted.current) return;

      console.log("🚀 Sending to backend...");
      // Send to backend
      const result = await ApiService.sendFrameForPrediction(photo.uri);

      // Cleanup
      cleanupImageUri(photo.uri);

      if (!isMounted.current) return;

      // Store prediction
      const newPrediction = {
        id: Date.now(),
        timestamp: new Date().toLocaleTimeString("tr-TR"),
        result: result,
      };

      setPredictions((prev) => [newPrediction, ...prev]);
      console.log("📸 Camera prediction result:", result);
    } catch (error) {
      if (!isMounted.current) return;

      console.error("❌ Camera capture error:", error);
      const appError =
        error instanceof AppError
          ? error
          : new AppError(MESSAGES.ERRORS.FRAME_CAPTURE, ERROR_TYPES.API, error);
      setError(appError);
      handleError(appError);
    } finally {
      if (isMounted.current) {
        setIsLoading(false);
      }
    }
  }, []); // Remove isCameraActive dependency// Start camera
  const startCamera = useCallback(async () => {
    console.log("🎥 Starting camera...");
    console.log("Current hasPermission:", permission?.granted);

    const hasPermissionGranted =
      permission?.granted || (await requestPermissions());
    console.log("hasPermissionGranted:", hasPermissionGranted);

    if (!hasPermissionGranted) {
      console.log("❌ No camera permission");
      return;
    }
    try {
      setError(null);
      console.log("✅ Setting camera active to true");
      setIsCameraActive(true);
      isCameraActiveRef.current = true; // Update ref as well
      setPredictions([]);

      // Start capturing frames at intervals
      console.log(
        "⏱️ Starting frame capture interval:",
        CONFIG.VIDEO.FRAME_CAPTURE_INTERVAL,
        "ms"
      );
      const id = setInterval(() => {
        console.log("⏰ Interval triggered - capturing frame...");
        captureAndSendFrame();
      }, CONFIG.VIDEO.FRAME_CAPTURE_INTERVAL);

      setIntervalId(id);
      console.log("✅ Camera started successfully with interval ID:", id);
    } catch (error) {
      console.error("❌ Start camera error:", error);
      const appError = new AppError(
        "Kamera başlatılamadı",
        ERROR_TYPES.UNKNOWN,
        error
      );
      setError(appError);
      handleError(appError);
    }
  }, [permission?.granted, requestPermissions, captureAndSendFrame]);
  // Stop camera
  const stopCamera = useCallback(async () => {
    try {
      setError(null);
      setIsCameraActive(false);
      isCameraActiveRef.current = false; // Update ref as well
      setIsLoading(false);

      // Clear interval
      if (intervalId) {
        clearInterval(intervalId);
        setIntervalId(null);
      }
    } catch (error) {
      console.error("Stop camera error:", error);
    }
  }, [intervalId]); // Clear predictions
  const clearPredictions = useCallback(async () => {
    setPredictions([]);
  }, []);

  return {
    hasPermission: permission?.granted,
    isCameraActive,
    isLoading,
    error,
    predictions,
    cameraRef,
    startCamera,
    stopCamera,
    clearPredictions,
    requestPermissions,
  };
};
