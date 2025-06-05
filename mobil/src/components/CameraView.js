import React from "react";
import { View, Text, StyleSheet } from "react-native";
import { CameraView as Camera } from "expo-camera";

const CameraView = ({
  cameraRef,
  hasPermission,
  isCameraActive,
  onRequestPermission,
}) => {
  if (hasPermission === null) {
    return (
      <View style={[styles.container, styles.centerContent]}>
        <Text style={styles.text}>Kamera izni kontrol ediliyor...</Text>
      </View>
    );
  }

  if (hasPermission === false) {
    return (
      <View style={[styles.container, styles.centerContent]}>
        <Text style={styles.text}>Kamera erişimi reddedildi</Text>
        <Text style={styles.subText}>
          Lütfen ayarlardan kamera iznini etkinleştirin
        </Text>
      </View>
    );
  }

  if (!isCameraActive) {
    return (
      <View style={[styles.container, styles.centerContent]}>
        <Text style={styles.text}>Kamera başlatılmadı</Text>
        <Text style={styles.subText}>
          Kamera butonuna basarak başlatabilirsiniz
        </Text>
      </View>
    );
  }
  return (
    <View style={styles.container}>
      <Camera ref={cameraRef} style={styles.camera} facing="back">
        <View style={styles.overlay}>
          <View style={styles.frameIndicator} />
        </View>
      </Camera>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    width: "100%",
    height: 300,
    backgroundColor: "#000",
    borderRadius: 8,
    overflow: "hidden",
  },
  centerContent: {
    justifyContent: "center",
    alignItems: "center",
    padding: 20,
  },
  camera: {
    flex: 1,
  },
  overlay: {
    flex: 1,
    backgroundColor: "transparent",
    justifyContent: "center",
    alignItems: "center",
  },
  frameIndicator: {
    width: 200,
    height: 200,
    borderWidth: 2,
    borderColor: "#00ff00",
    backgroundColor: "transparent",
    borderRadius: 8,
  },
  text: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "bold",
    textAlign: "center",
    marginBottom: 8,
  },
  subText: {
    color: "#ccc",
    fontSize: 14,
    textAlign: "center",
  },
});

export default CameraView;
