/**
 * Navigation utility functions for handling URL paths in different environments
 */

/**
 * Helper function to generate correct URLs accounting for base path in production
 * @param path - The path to append to the base URL
 * @returns The complete URL with appropriate base path for the current environment
 */
export const getNavUrl = (path: string): string => {
  // Check if we're in production using NODE_ENV
  const isProduction = process.env.NODE_ENV === 'production';
  const basePath = isProduction ? '/GeoAuPredict' : '';
  return `${basePath}${path}`;
};
