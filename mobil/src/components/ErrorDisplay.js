import { View, Text, TouchableOpacity } from "react-native";
import PropTypes from "prop-types";
import { componentStyles } from "../styles/styles";

const ErrorDisplay = ({ error, onDismiss, style }) => {
  if (!error) return null;

  return (
    <View style={[componentStyles.errorContainer, style]}>
      <View style={componentStyles.errorContent}>
        <Text style={componentStyles.errorTitle}>Hata</Text>
        <Text style={componentStyles.errorMessage}>{error.message}</Text>
        {onDismiss && (
          <TouchableOpacity
            style={componentStyles.errorDismissButton}
            onPress={onDismiss}
            activeOpacity={0.7}
          >
            <Text style={componentStyles.errorDismissText}>Tamam</Text>
          </TouchableOpacity>
        )}
      </View>
    </View>
  );
};

ErrorDisplay.propTypes = {
  error: PropTypes.shape({
    message: PropTypes.string.isRequired,
    type: PropTypes.string,
  }),
  onDismiss: PropTypes.func,
  style: PropTypes.oneOfType([PropTypes.object, PropTypes.array]),
};

export default ErrorDisplay;
