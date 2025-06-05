import { useAppContext } from "../context/AppContext";
import * as ImagePicker from "expo-image-picker";
import * as VideoThumbnails from "expo-video-thumbnails";
import * as ImageManipulator from "expo-image-manipulator";
import { CONFIG, MESSAGES } from "../constants/config";
import {
  ApiService,
  handleError,
  AppError,
  ERROR_TYPES,
  StorageManager,
  calculateOptimalImageSize,
  cleanupImageUri,
} from "../utils";
import { useThrottle, useMountedRef } from "./usePerformance";
import { useState, useEffect, useCallback, useRef } from "react";
import { Alert } from "react-native";

export const useVideoPlayer = () => {
  const [videoUri, setVideoUri] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [predictions, setPredictions] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [intervalId, setIntervalId] = useState(null);
  const [error, setError] = useState(null);

  const videoRef = useRef(null);
  const isMounted = useMountedRef();
  const { settings } = useAppContext();

  // Throttled capture function for performance
  const throttledCapture = useThrottle(async () => {
    await captureAndSendFrame();
  }, CONFIG.VIDEO.FRAME_CAPTURE_INTERVAL);

  // Load predictions on mount
  useEffect(() => {
    loadStoredPredictions();
  }, []);

  // Save predictions when they change
  useEffect(() => {
    if (predictions.length > 0) {
      savePredictions();
    }
  }, [predictions]);

  const loadStoredPredictions = useCallback(async () => {
    try {
      const storedPredictions = await StorageManager.loadPredictionsHistory();
      setPredictions(storedPredictions);
    } catch (error) {
      handleError(
        new AppError(
          "Kaydedilmiş veriler yüklenemedi",
          ERROR_TYPES.UNKNOWN,
          error
        ),
        false
      );
    }
  }, []);

  const savePredictions = useCallback(async () => {
    try {
      await StorageManager.savePredictionsHistory(predictions);
    } catch (error) {
      console.error("Failed to save predictions:", error);
    }
  }, [predictions]);
  // Request permissions and select video
  const selectVideo = useCallback(async () => {
    try {
      setError(null);
      const { status } =
        await ImagePicker.requestMediaLibraryPermissionsAsync();

      if (status !== "granted") {
        throw new AppError(
          MESSAGES.ERRORS.PERMISSION_DENIED,
          ERROR_TYPES.PERMISSION
        );
      }

      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Videos,
        allowsEditing: false,
        quality: 1,
      });
      if (!result.canceled && result.assets[0]) {
        setVideoUri(result.assets[0].uri);
        setPredictions([]);

        // Reset video state when new video is selected
        setIsPlaying(false);
        setIsLoading(false);

        // Clear previous interval if exists
        if (intervalId) {
          clearInterval(intervalId);
          setIntervalId(null);
        }
      }
    } catch (error) {
      const appError =
        error instanceof AppError
          ? error
          : new AppError(
              MESSAGES.ERRORS.VIDEO_SELECTION,
              ERROR_TYPES.UNKNOWN,
              error
            );
      setError(appError);
      handleError(appError);
    }
  }, [intervalId]); // Capture frame and send to API
  const captureAndSendFrame = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);

      const status = await videoRef.current.getStatusAsync();

      // Check if video is actually playing based on status, not just our state
      if (!status.isPlaying) {
        setIsLoading(false);
        return;
      }

      const currentTime = status.positionMillis || 0;
      const { uri: thumbnailUri } = await VideoThumbnails.getThumbnailAsync(
        videoUri,
        {
          time: currentTime,
          quality: CONFIG.VIDEO.THUMBNAIL_QUALITY,
        }
      );
      const optimalSize = calculateOptimalImageSize(
        CONFIG.VIDEO.THUMBNAIL_WIDTH,
        CONFIG.VIDEO.THUMBNAIL_WIDTH,
        CONFIG.VIDEO.THUMBNAIL_WIDTH
      );
      const manipResult = await ImageManipulator.manipulateAsync(
        thumbnailUri,
        [{ resize: optimalSize }],
        {
          compress: CONFIG.VIDEO.COMPRESSION_QUALITY,
          format: ImageManipulator.SaveFormat.JPEG,
        }
      );

      // Only proceed if component is still mounted
      if (!isMounted.current) return;

      const result = await ApiService.sendFrameForPrediction(manipResult.uri);

      // Cleanup temporary image
      cleanupImageUri(manipResult.uri);
      cleanupImageUri(thumbnailUri);

      if (!isMounted.current) return;
      const newPrediction = {
        id: Date.now(),
        timestamp: new Date().toLocaleTimeString("tr-TR"),
        result: result,
      };
      setPredictions((prev) => [newPrediction, ...prev]);
    } catch (error) {
      if (!isMounted.current) return;

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
  }, [videoUri, isMounted, predictions]);

  const startPlayback = useCallback(async () => {
    if (!videoUri) {
      const appError = new AppError(
        MESSAGES.ERRORS.NO_VIDEO_SELECTED,
        ERROR_TYPES.VALIDATION
      );
      setError(appError);
      handleError(appError);
      return;
    }

    try {
      setError(null);
      await videoRef.current?.playAsync();
      setIsPlaying(true);

      const id = setInterval(
        throttledCapture,
        CONFIG.VIDEO.FRAME_CAPTURE_INTERVAL
      );
      setIntervalId(id);
      setTimeout(() => {
        captureAndSendFrame();
      }, 500);
    } catch (error) {
      const appError = new AppError(
        MESSAGES.ERRORS.VIDEO_PLAYBACK,
        ERROR_TYPES.UNKNOWN,
        error
      );
      setError(appError);
      handleError(appError);
    }
  }, [videoUri, throttledCapture, captureAndSendFrame]);
  // Stop video playback
  const stopPlayback = useCallback(async () => {
    try {
      setError(null);
      await videoRef.current?.pauseAsync();
      setIsPlaying(false);

      // Clear interval to stop sending frames to backend
      if (intervalId) {
        clearInterval(intervalId);
        setIntervalId(null);
      }

      // Stop loading state immediately when video is paused
      setIsLoading(false);
    } catch (error) {
      console.error("Stop playback error:", error);
    }
  }, [intervalId]);
  // Clear all predictions
  const clearPredictions = useCallback(async () => {
    setPredictions([]);
    await StorageManager.savePredictionsHistory([]);
  }, []);

  // Cleanup interval on unmount
  useEffect(() => {
    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [intervalId]);
  return {
    videoUri,
    isPlaying,
    predictions,
    isLoading,
    error,
    videoRef,
    selectVideo,
    startPlayback,
    stopPlayback,
    clearPredictions,
  };
};
