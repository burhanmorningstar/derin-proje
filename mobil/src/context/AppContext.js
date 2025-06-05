import React, { createContext, useContext, useReducer, useEffect } from "react";
import PropTypes from "prop-types";
import { StorageManager } from "../utils";
import { CONFIG } from "../constants/config";

// Initial state - sync with CONFIG
const initialState = {
  theme: "light",
  frameInterval: CONFIG.VIDEO.FRAME_CAPTURE_INTERVAL,
  apiEndpoint: CONFIG.API.BASE_URL,
  autoSave: true,
  notifications: true,
  compressionQuality: CONFIG.VIDEO.COMPRESSION_QUALITY,
  thumbnailQuality: CONFIG.VIDEO.THUMBNAIL_QUALITY,
};

// Action types
const ACTION_TYPES = {
  SET_THEME: "SET_THEME",
  SET_FRAME_INTERVAL: "SET_FRAME_INTERVAL",
  SET_API_ENDPOINT: "SET_API_ENDPOINT",
  SET_AUTO_SAVE: "SET_AUTO_SAVE",
  SET_NOTIFICATIONS: "SET_NOTIFICATIONS",
  SET_COMPRESSION_QUALITY: "SET_COMPRESSION_QUALITY",
  SET_THUMBNAIL_QUALITY: "SET_THUMBNAIL_QUALITY",
  LOAD_SETTINGS: "LOAD_SETTINGS",
  RESET_SETTINGS: "RESET_SETTINGS",
};

// Reducer
const settingsReducer = (state, action) => {
  switch (action.type) {
    case ACTION_TYPES.SET_THEME:
      return { ...state, theme: action.payload };
    case ACTION_TYPES.SET_FRAME_INTERVAL:
      return { ...state, frameInterval: action.payload };
    case ACTION_TYPES.SET_API_ENDPOINT:
      return { ...state, apiEndpoint: action.payload };
    case ACTION_TYPES.SET_AUTO_SAVE:
      return { ...state, autoSave: action.payload };
    case ACTION_TYPES.SET_NOTIFICATIONS:
      return { ...state, notifications: action.payload };
    case ACTION_TYPES.SET_COMPRESSION_QUALITY:
      return { ...state, compressionQuality: action.payload };
    case ACTION_TYPES.SET_THUMBNAIL_QUALITY:
      return { ...state, thumbnailQuality: action.payload };
    case ACTION_TYPES.LOAD_SETTINGS:
      return { ...state, ...action.payload };
    case ACTION_TYPES.RESET_SETTINGS:
      return { ...initialState };
    default:
      return state;
  }
};

// Create context
const AppContext = createContext();

// Provider component
export const AppProvider = ({ children }) => {
  const [settings, dispatch] = useReducer(settingsReducer, initialState);

  // Load settings on mount
  useEffect(() => {
    loadSettings();
  }, []);

  // Save settings when they change
  useEffect(() => {
    if (settings.autoSave) {
      saveSettings();
    }
  }, [settings]);

  const loadSettings = async () => {
    try {
      const savedSettings = await StorageManager.loadAppSettings();
      if (Object.keys(savedSettings).length > 0) {
        dispatch({ type: ACTION_TYPES.LOAD_SETTINGS, payload: savedSettings });
      }
    } catch (error) {
      console.error("Failed to load settings:", error);
    }
  };

  const saveSettings = async () => {
    try {
      await StorageManager.saveAppSettings(settings);
    } catch (error) {
      console.error("Failed to save settings:", error);
    }
  };

  const updateSetting = (type, value) => {
    dispatch({ type, payload: value });
  };

  const resetSettings = () => {
    dispatch({ type: ACTION_TYPES.RESET_SETTINGS });
  };

  const value = {
    settings,
    updateSetting,
    resetSettings,
    ACTION_TYPES,
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
};

AppProvider.propTypes = {
  children: PropTypes.node.isRequired,
};

// Custom hook
export const useAppContext = () => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error("useAppContext must be used within an AppProvider");
  }
  return context;
};

export default AppContext;
