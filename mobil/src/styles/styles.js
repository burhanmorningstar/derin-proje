import { StyleSheet } from "react-native";
import { CONFIG } from "../constants/config";

const { COLORS, SPACING, BORDER_RADIUS, FONT_SIZES } = CONFIG.UI;

export const globalStyles = StyleSheet.create({
  // App-specific styles
  safeArea: {
    flex: 1,
    backgroundColor: COLORS.BACKGROUND,
  },

  appContainer: {
    flex: 1,
    backgroundColor: "#FFFFFF", // Beyaz arka plan - uygulama içeriği için
  },

  // Shadows
  shadowSmall: {
    shadowColor: COLORS.SHADOW,
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },

  shadowMedium: {
    shadowColor: COLORS.SHADOW,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.15,
    shadowRadius: 4,
    elevation: 4,
  },

  shadowLarge: {
    shadowColor: COLORS.SHADOW,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 8,
    elevation: 8,
  },

  // Typography
  textPrimary: {
    color: COLORS.TEXT_PRIMARY,
  },

  textSecondary: {
    color: COLORS.TEXT_SECONDARY,
  },

  textCenter: {
    textAlign: "center",
  },

  textBold: {
    fontWeight: "bold",
  },

  // Spacing
  marginXS: { margin: SPACING.XS },
  marginSM: { margin: SPACING.SM },
  marginMD: { margin: SPACING.MD },
  marginLG: { margin: SPACING.LG },
  marginXL: { margin: SPACING.XL },

  paddingXS: { padding: SPACING.XS },
  paddingSM: { padding: SPACING.SM },
  paddingMD: { padding: SPACING.MD },
  paddingLG: { padding: SPACING.LG },
  paddingXL: { padding: SPACING.XL },

  // Flex
  flexCenter: {
    justifyContent: "center",
    alignItems: "center",
  },

  flexRow: {
    flexDirection: "row",
  },

  flexBetween: {
    justifyContent: "space-between",
  },

  flexAround: {
    justifyContent: "space-around",
  },

  // Border radius
  roundedSM: { borderRadius: BORDER_RADIUS.SM },
  roundedMD: { borderRadius: BORDER_RADIUS.MD },
  roundedLG: { borderRadius: BORDER_RADIUS.LG },
  roundedXL: { borderRadius: BORDER_RADIUS.XL },
});

export const componentStyles = StyleSheet.create({
  // Header
  header: {
    paddingHorizontal: SPACING.LG,
    paddingVertical: SPACING.MD,
    backgroundColor: COLORS.HEADER,
    ...globalStyles.shadowMedium,
  },

  headerTitle: {
    fontSize: FONT_SIZES.XXL,
    fontWeight: "bold",
    color: "white",
    textAlign: "center",
  },
  // Video
  videoContainer: {
    height: 220,
    margin: SPACING.SM,
    backgroundColor: "black",
    borderRadius: BORDER_RADIUS.LG,
    overflow: "hidden",
    ...globalStyles.shadowLarge,
  },

  video: {
    flex: 1,
  },

  placeholderContainer: {
    height: 220,
    margin: SPACING.LG,
    backgroundColor: COLORS.BORDER,
    borderRadius: BORDER_RADIUS.LG,
    borderWidth: 2,
    borderStyle: "dashed",
    borderColor: COLORS.TEXT_SECONDARY,
    ...globalStyles.flexCenter,
  },

  placeholderText: {
    fontSize: FONT_SIZES.LG,
    color: COLORS.TEXT_SECONDARY,
    fontWeight: "500",
  }, // Controls
  controlsContainer: {
    paddingHorizontal: SPACING.LG,
    paddingVertical: SPACING.MD,
    gap: SPACING.MD,
    zIndex: 5, // Ensure controls stay on top
    backgroundColor: "white", // Add background to prevent overlap
    elevation: 5, // Android elevation
  },

  selectionButtonsContainer: {
    flexDirection: "row",
    justifyContent: "space-between",
    gap: SPACING.MD,
    marginBottom: SPACING.SM,
  },

  halfButton: {
    flex: 1,
    maxWidth: "48%",
  },
  playButton: {
    alignSelf: "center",
    minWidth: 120,
    flex: 0, // Override flex from button base style
    maxWidth: 200,
    zIndex: 10, // Ensure it's above other elements
  },

  button: {
    backgroundColor: COLORS.PRIMARY,
    paddingHorizontal: SPACING.LG,
    paddingVertical: SPACING.MD,
    borderRadius: BORDER_RADIUS.MD,
    maxWidth: 150,
    ...globalStyles.flexCenter,
    ...globalStyles.shadowMedium,
  },

  buttonSecondary: {
    backgroundColor: COLORS.SECONDARY,
  },

  buttonSuccess: {
    backgroundColor: COLORS.SUCCESS,
  },

  buttonDanger: {
    backgroundColor: COLORS.DANGER,
  },

  buttonDisabled: {
    backgroundColor: COLORS.TEXT_SECONDARY,
    opacity: 0.6,
  },
  buttonText: {
    color: "white",
    fontSize: FONT_SIZES.LG,
    fontWeight: "700", // Make text bolder
    textAlign: "center",
    zIndex: 15, // Ensure text stays visible
  },

  // Loading
  loadingContainer: {
    flexDirection: "row",
    justifyContent: "center",
    alignItems: "center",
    paddingVertical: SPACING.MD,
    backgroundColor: COLORS.CARD_BACKGROUND,
    marginHorizontal: SPACING.LG,
    borderRadius: BORDER_RADIUS.MD,
    marginBottom: SPACING.SM,
    ...globalStyles.shadowSmall,
  },

  loadingText: {
    marginLeft: SPACING.SM,
    fontSize: FONT_SIZES.MD,
    color: COLORS.TEXT_SECONDARY,
    fontWeight: "500",
  },

  // Predictions
  predictionsContainer: {
    flex: 1,
    margin: SPACING.LG,
    backgroundColor: COLORS.CARD_BACKGROUND,
    borderRadius: BORDER_RADIUS.LG,
    padding: SPACING.LG,
    ...globalStyles.shadowMedium,
  },
  predictionsHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: SPACING.MD,
    paddingBottom: SPACING.SM,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.BORDER,
  },

  predictionsHeaderActions: {
    flexDirection: "row",
    alignItems: "center",
    gap: SPACING.SM,
  },

  predictionsTitle: {
    fontSize: FONT_SIZES.XL,
    fontWeight: "bold",
    color: COLORS.TEXT_PRIMARY,
  },
  predictionsCount: {
    backgroundColor: COLORS.PRIMARY,
    color: "white",
    fontSize: FONT_SIZES.SM,
    paddingHorizontal: SPACING.SM,
    paddingVertical: SPACING.XS,
    borderRadius: BORDER_RADIUS.SM,
    fontWeight: "bold",
    overflow: "hidden",
  },

  clearButton: {
    backgroundColor: COLORS.DANGER,
    paddingHorizontal: SPACING.SM,
    paddingVertical: SPACING.XS,
    borderRadius: BORDER_RADIUS.SM,
  },

  clearButtonText: {
    color: "white",
    fontSize: FONT_SIZES.SM,
    fontWeight: "600",
  },

  predictionsList: {
    flex: 1,
  },
  // Prediction Item
  predictionItem: {
    backgroundColor: COLORS.BACKGROUND,
    padding: SPACING.MD,
    marginBottom: SPACING.SM,
    borderRadius: BORDER_RADIUS.MD,
    ...globalStyles.shadowSmall,
  },

  predictionHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: SPACING.XS,
  },

  predictionTime: {
    fontSize: FONT_SIZES.SM,
    color: COLORS.TEXT_SECONDARY,
    fontWeight: "500",
  },

  predictionLabel: {
    fontSize: FONT_SIZES.XL,
    fontWeight: "bold",
    marginBottom: SPACING.XS,
  },

  predictionLabelViolence: {
    color: COLORS.VIOLENCE,
  },

  predictionLabelNonViolence: {
    color: COLORS.NON_VIOLENCE,
  },
  predictionConfidence: {
    fontSize: FONT_SIZES.LG,
    fontWeight: "600",
    marginBottom: SPACING.XS,
  },

  // New styles for violence percentage and confidence score
  violencePercentage: {
    fontSize: FONT_SIZES.XXXL,
    fontWeight: "900",
    textAlign: "center",
    marginVertical: SPACING.SM,
    letterSpacing: 1,
  },

  confidenceScore: {
    fontSize: FONT_SIZES.MD,
    fontWeight: "500",
    textAlign: "center",
    color: COLORS.TEXT_SECONDARY,
    marginBottom: SPACING.SM,
  },

  predictionProbabilities: {
    fontSize: FONT_SIZES.SM,
    color: COLORS.TEXT_SECONDARY,
    marginBottom: SPACING.SM,
  },

  noPredictions: {
    textAlign: "center",
    color: COLORS.TEXT_SECONDARY,
    fontStyle: "italic",
    marginTop: SPACING.LG,
    fontSize: FONT_SIZES.LG,
  },

  // Error Display
  errorContainer: {
    backgroundColor: "rgba(220, 53, 69, 0.1)",
    marginHorizontal: SPACING.LG,
    marginVertical: SPACING.SM,
    borderRadius: BORDER_RADIUS.MD,
    borderWidth: 1,
    borderColor: COLORS.DANGER,
  },

  errorContent: {
    padding: SPACING.MD,
  },

  errorTitle: {
    fontSize: FONT_SIZES.LG,
    fontWeight: "bold",
    color: COLORS.DANGER,
    marginBottom: SPACING.XS,
  },

  errorMessage: {
    fontSize: FONT_SIZES.MD,
    color: COLORS.TEXT_PRIMARY,
    marginBottom: SPACING.SM,
    lineHeight: 20,
  },

  errorDismissButton: {
    backgroundColor: COLORS.DANGER,
    paddingHorizontal: SPACING.MD,
    paddingVertical: SPACING.SM,
    borderRadius: BORDER_RADIUS.SM,
    alignSelf: "flex-end",
  },

  errorDismissText: {
    color: "white",
    fontSize: FONT_SIZES.SM,
    fontWeight: "600",
  },
});
