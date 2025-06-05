import { View, Text } from "react-native";
import { Video, ResizeMode } from "expo-av";
import PropTypes from "prop-types";
import { componentStyles } from "../styles/styles";
import { MESSAGES } from "../constants/config";

const VideoPlayer = ({ videoUri, videoRef, isPlaying }) => {
  if (!videoUri) {
    return (
      <View style={componentStyles.placeholderContainer}>
        <Text style={componentStyles.placeholderText}>
          {MESSAGES.INFO.SELECT_VIDEO}
        </Text>
      </View>
    );
  }

  return (
    <View style={componentStyles.videoContainer}>
      <Video
        ref={videoRef}
        source={{ uri: videoUri }}
        style={componentStyles.video}
        useNativeControls
        resizeMode={ResizeMode.CONTAIN}
        shouldPlay={false}
        isLooping
      />
    </View>
  );
};

VideoPlayer.propTypes = {
  videoUri: PropTypes.string,
  videoRef: PropTypes.object.isRequired,
  isPlaying: PropTypes.bool.isRequired,
};

export default VideoPlayer;
