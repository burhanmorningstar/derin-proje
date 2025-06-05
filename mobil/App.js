import React, { useState } from "react";
import { View } from "react-native";
import { SafeAreaProvider, SafeAreaView } from "react-native-safe-area-context";
import {
  Header,
  VideoPlayer,
  VideoControls,
  LoadingIndicator,
  PredictionsList,
  ErrorDisplay,
  CameraView,
} from "./src/components";
import { useVideoPlayer, useCamera } from "./src/hooks";
import { AppProvider } from "./src/context/AppContext";
import { globalStyles } from "./src/styles/styles";
import { MESSAGES } from "./src/constants/config";

const AppContent = () => {
  const [displayError, setDisplayError] = useState(null);

  // Video player hook
  const {
    videoUri,
    isPlaying,
    predictions: videoPredictions,
    isLoading: videoLoading,
    error: videoError,
    videoRef,
    selectVideo,
    startPlayback,
    stopPlayback,
    clearPredictions: clearVideoPredictions,
  } = useVideoPlayer();
  // Camera hook
  const {
    hasPermission,
    isCameraActive,
    isLoading: cameraLoading,
    error: cameraError,
    predictions: cameraPredictions,
    cameraRef,
    startCamera,
    stopCamera,
    clearPredictions: clearCameraPredictions,
    requestPermissions,
  } = useCamera();
  // Combined states
  const isLoading = videoLoading || cameraLoading;
  const error = videoError || cameraError;
  const predictions = isCameraActive ? cameraPredictions : videoPredictions;

  // Combined clear function
  const clearPredictions = () => {
    if (isCameraActive) {
      clearCameraPredictions();
    } else {
      clearVideoPredictions();
    }
  };

  // Handle error display
  React.useEffect(() => {
    if (error) {
      setDisplayError(error);
    }
  }, [error]);

  const handleDismissError = () => {
    setDisplayError(null);
  };
  return (
    <SafeAreaView style={globalStyles.safeArea} edges={["top"]}>
      <View style={globalStyles.appContainer}>
        <Header title="Video Şiddet Analizi" />
        <ErrorDisplay error={displayError} onDismiss={handleDismissError} />
        {isCameraActive ? (
          <CameraView
            cameraRef={cameraRef}
            hasPermission={hasPermission}
            isCameraActive={isCameraActive}
            onRequestPermission={requestPermissions}
          />
        ) : (
          <VideoPlayer
            videoUri={videoUri}
            videoRef={videoRef}
            isPlaying={isPlaying}
          />
        )}
        <VideoControls
          videoUri={videoUri}
          isPlaying={isPlaying}
          isCameraActive={isCameraActive}
          onSelectVideo={selectVideo}
          onStartPlayback={startPlayback}
          onStopPlayback={stopPlayback}
          onStartCamera={startCamera}
          onStopCamera={stopCamera}
        />
        {isLoading && (
          <LoadingIndicator message={MESSAGES.INFO.SENDING_TO_API} />
        )}
        <PredictionsList
          predictions={predictions}
          onClearPredictions={clearPredictions}
        />
      </View>
    </SafeAreaView>
  );
};

export default function App() {
  return (
    <SafeAreaProvider>
      <AppProvider>
        <AppContent />
      </AppProvider>
    </SafeAreaProvider>
  );
}
