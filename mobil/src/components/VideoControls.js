import { View } from "react-native";
import PropTypes from "prop-types";
import { componentStyles } from "../styles/styles";
import Button from "./Button";

const VideoControls = ({
  videoUri,
  isPlaying,
  onSelectVideo,
  onStartPlayback,
  onStopPlayback,
  // Camera props
  isCameraActive,
  onStartCamera,
  onStopCamera,
}) => {
  return (
    <View style={componentStyles.controlsContainer}>
      {/* Video ve Kamera seçim butonları */}
      <View style={componentStyles.selectionButtonsContainer}>
        <Button
          title="Video Seç"
          onPress={onSelectVideo}
          variant="secondary"
          style={componentStyles.halfButton}
        />
        <Button
          title="Kamera"
          onPress={isCameraActive ? onStopCamera : onStartCamera}
          variant={isCameraActive ? "danger" : "primary"}
          style={componentStyles.halfButton}
        />
      </View>

      {/* Oynatma kontrolü */}
      <Button
        title={isPlaying || isCameraActive ? "Durdur" : "Başlat"}
        onPress={
          isCameraActive
            ? onStopCamera
            : isPlaying
            ? onStopPlayback
            : onStartPlayback
        }
        disabled={!videoUri && !isCameraActive}
        variant={isPlaying || isCameraActive ? "danger" : "success"}
        style={componentStyles.playButton}
      />
    </View>
  );
};

VideoControls.propTypes = {
  videoUri: PropTypes.string,
  isPlaying: PropTypes.bool.isRequired,
  onSelectVideo: PropTypes.func.isRequired,
  onStartPlayback: PropTypes.func.isRequired,
  onStopPlayback: PropTypes.func.isRequired,
  isCameraActive: PropTypes.bool.isRequired,
  onStartCamera: PropTypes.func.isRequired,
  onStopCamera: PropTypes.func.isRequired,
};

export default VideoControls;
