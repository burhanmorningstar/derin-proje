import React from "react";
import { View, Text, ScrollView, TouchableOpacity } from "react-native";
import PropTypes from "prop-types";
import { componentStyles } from "../styles/styles";
import { MESSAGES } from "../constants/config";
import PredictionItem from "./PredictionItem";

const PredictionsList = ({ predictions, onClearPredictions }) => {
  return (
    <View style={componentStyles.predictionsContainer}>
      <View style={componentStyles.predictionsHeader}>
        <Text style={componentStyles.predictionsTitle}>Tahmin Sonuçları</Text>
        <View style={componentStyles.predictionsHeaderActions}>
          <Text style={componentStyles.predictionsCount}>
            {predictions.length}
          </Text>
          {predictions.length > 0 && onClearPredictions && (
            <TouchableOpacity
              style={componentStyles.clearButton}
              onPress={onClearPredictions}
              activeOpacity={0.7}
            >
              <Text style={componentStyles.clearButtonText}>Temizle</Text>
            </TouchableOpacity>
          )}
        </View>
      </View>

      <ScrollView
        style={componentStyles.predictionsList}
        showsVerticalScrollIndicator={false}
        contentContainerStyle={{ paddingBottom: 20 }}
      >
        {predictions.length > 0 ? (
          predictions.map((prediction) => (
            <PredictionItem key={prediction.id} prediction={prediction} />
          ))
        ) : (
          <Text style={componentStyles.noPredictions}>
            {MESSAGES.INFO.NO_PREDICTIONS}
          </Text>
        )}
      </ScrollView>
    </View>
  );
};

PredictionsList.propTypes = {
  predictions: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.number.isRequired,
      timestamp: PropTypes.string.isRequired,
      result: PropTypes.object.isRequired,
    })
  ).isRequired,
  onClearPredictions: PropTypes.func,
};

export default PredictionsList;
