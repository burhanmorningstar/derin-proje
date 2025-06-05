import React from "react";
import { View, Text, StatusBar } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import PropTypes from "prop-types";
import { componentStyles, globalStyles } from "../styles/styles";

const Header = ({ title = "" }) => {
  const insets = useSafeAreaInsets();

  return (
    <>
      <StatusBar barStyle="light-content" backgroundColor="#007AFF" />
      <View style={[componentStyles.header]}>
        <Text style={componentStyles.headerTitle}>{title}</Text>
      </View>
    </>
  );
};

Header.propTypes = {
  title: PropTypes.string,
};

export default Header;
