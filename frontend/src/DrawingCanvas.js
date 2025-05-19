import React, { useRef } from "react";
import { View, Button, StyleSheet } from "react-native";
import Signature from "react-native-signature-canvas";

export default function DrawingCanvas({ onPredict }) {
  const sigRef = useRef(null);

  const handleOK = (base64) => {
    // The signature canvas may return data URLs that already include the header
    // Make sure we're dealing with just the base64 data for API submission
    const cleanBase64 = base64.includes("data:image")
      ? base64.split(",")[1]
      : base64;

    // Remove any non-base64 characters that might cause issues
    const validBase64 = cleanBase64.replace(/[^A-Za-z0-9+/=]/g, "");

    // Add the header to the base64 string to make it a valid image
    const formattedBase64 = `data:image/png;base64,${validBase64}`;
    onPredict(formattedBase64);
  };

  return (
    <View style={{ flex: 1 }}>
      <Signature
        onOK={handleOK} // Use the handleOK function to format the base64 data
        ref={sigRef}
        penColor="black"
        backgroundColor="white"
        descriptionText=""
        webStyle="body,html{width:100%;height:100%;margin:0;}"
      />
      <View style={styles.buttons}>
        <Button
          title="Predict"
          onPress={() => sigRef.current.readSignature()}
        />

        <Button title="Reset" onPress={() => sigRef.current.clearSignature()} />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  buttons: {
    position: "absolute",
    bottom: 40,
    flexDirection: "row",
    gap: 20,
    alignSelf: "center",
  },
});
