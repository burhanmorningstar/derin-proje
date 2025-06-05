import { View, Text } from "react-native";
import PropTypes from "prop-types";
import { componentStyles } from "../styles/styles";
import { CONFIG } from "../constants/config";

const PredictionItem = ({ prediction }) => {
  const { result, timestamp } = prediction;
  const predLabel = result?.prediction || "";
  const violenceProb = result?.violence_prob ?? 0;
  const nonviolenceProb = result?.nonviolence_prob ?? 0;

  const isViolence = predLabel === "Violence";
  const confidence = isViolence
    ? Math.round(violenceProb * 100)
    : Math.round(nonviolenceProb * 100);

  // Dynamic card styling based on violence detection
  const itemStyle = [
    componentStyles.predictionItem,
    {
      backgroundColor: isViolence
        ? CONFIG.UI.COLORS.VIOLENCE + "20" // Light red background for violence
        : CONFIG.UI.COLORS.NON_VIOLENCE + "20", // Light green background for non-violence
      borderLeftWidth: 4,
      borderLeftColor: isViolence
        ? CONFIG.UI.COLORS.VIOLENCE
        : CONFIG.UI.COLORS.NON_VIOLENCE,
    },
  ];
  const labelStyle = [
    componentStyles.predictionLabel,
    isViolence
      ? componentStyles.predictionLabelViolence
      : componentStyles.predictionLabelNonViolence,
  ];

  return (
    <View style={itemStyle}>
      <View style={componentStyles.predictionHeader}>
        <Text style={componentStyles.predictionTime}>{timestamp}</Text>
      </View>

      <Text style={labelStyle}>
        {isViolence ? "ŞİDDET TESPİT EDİLDİ" : "ŞİDDET TESPİT EDİLMEDİ"}
      </Text>

      <Text
        style={[
          componentStyles.violencePercentage,
          {
            color: isViolence
              ? CONFIG.UI.COLORS.VIOLENCE
              : CONFIG.UI.COLORS.NON_VIOLENCE,
          },
        ]}
      >
        %{Math.round(violenceProb * 100)}
      </Text>
    </View>
  );
};

PredictionItem.propTypes = {
  prediction: PropTypes.shape({
    id: PropTypes.number.isRequired,
    timestamp: PropTypes.string.isRequired,
    result: PropTypes.object.isRequired,
  }).isRequired,
};

export default PredictionItem;
