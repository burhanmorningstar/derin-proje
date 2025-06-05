import React from "react";
import { View, Text, ActivityIndicator } from "react-native";
import PropTypes from "prop-types";
import { componentStyles } from "../styles/styles";
import { CONFIG } from "../constants/config";

const LoadingIndicator = ({
  message = "Yükleniyor...",
  size = "small",
  color = CONFIG.UI.COLORS.PRIMARY,
  style,
}) => {
  return (
    <View style={[componentStyles.loadingContainer, style]}>
      <ActivityIndicator size={size} color={color} />
      <Text style={componentStyles.loadingText}>{message}</Text>
    </View>
  );
};

LoadingIndicator.propTypes = {
  message: PropTypes.string,
  size: PropTypes.oneOf(["small", "large"]),
  color: PropTypes.string,
  style: PropTypes.oneOfType([PropTypes.object, PropTypes.array]),
};

export default LoadingIndicator;
