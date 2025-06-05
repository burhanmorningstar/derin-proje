import React, { useCallback, useRef } from "react";

// Debounce hook for performance optimization
export const useDebounce = (callback, delay) => {
  const timeoutRef = useRef(null);

  return useCallback(
    (...args) => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }

      timeoutRef.current = setTimeout(() => {
        callback(...args);
      }, delay);
    },
    [callback, delay]
  );
};

// Throttle hook for limiting function calls
export const useThrottle = (callback, delay) => {
  const lastCallRef = useRef(0);

  return useCallback(
    (...args) => {
      const now = Date.now();
      if (now - lastCallRef.current >= delay) {
        lastCallRef.current = now;
        callback(...args);
      }
    },
    [callback, delay]
  );
};

// Previous value hook for comparison
export const usePrevious = (value) => {
  const ref = useRef();

  React.useEffect(() => {
    ref.current = value;
  });

  return ref.current;
};

// Mount status hook
export const useMountedRef = () => {
  const mountedRef = useRef(false);

  React.useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  return mountedRef;
};
