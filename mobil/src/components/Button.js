import React from "react";
import { View, Text, TouchableOpacity } from "react-native";
import PropTypes from "prop-types";
import { componentStyles } from "../styles/styles";

const Button = ({
  title,
  onPress,
  disabled = false,
  variant = "primary",
  style,
  textStyle,
  ...props
}) => {
  const getButtonStyle = () => {
    const baseStyle = [componentStyles.button];

    switch (variant) {
      case "secondary":
        baseStyle.push(componentStyles.buttonSecondary);
        break;
      case "success":
        baseStyle.push(componentStyles.buttonSuccess);
        break;
      case "danger":
        baseStyle.push(componentStyles.buttonDanger);
        break;
      default:
        // primary is default
        break;
    }

    if (disabled) {
      baseStyle.push(componentStyles.buttonDisabled);
    }

    if (style) {
      baseStyle.push(style);
    }

    return baseStyle;
  };

  return (
    <TouchableOpacity
      style={getButtonStyle()}
      onPress={onPress}
      disabled={disabled}
      activeOpacity={0.7}
      {...props}
    >
      <Text style={[componentStyles.buttonText, textStyle]}>{title}</Text>
    </TouchableOpacity>
  );
};

Button.propTypes = {
  title: PropTypes.string.isRequired,
  onPress: PropTypes.func.isRequired,
  disabled: PropTypes.bool,
  variant: PropTypes.oneOf(["primary", "secondary", "success", "danger"]),
  style: PropTypes.oneOfType([PropTypes.object, PropTypes.array]),
  textStyle: PropTypes.oneOfType([PropTypes.object, PropTypes.array]),
};

export default Button;
