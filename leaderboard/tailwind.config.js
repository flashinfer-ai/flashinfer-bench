const colors = require('tailwindcss/colors');

/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./kernelboard/templates/**/*.html"],
  safelist: [ 
    'code-block-fade',
    'sr-only',
    'h-5',
    'w-5',
    'toast-error',
    'toast-error-content',
    'toast-error-close-btn',
    'toast-default',
    'toast-default-content',
    'toast-default-close-btn',
    'animate-error-toast-in',
    'animate-error-toast-out',
    'animate-default-toast-in',
    'animate-default-toast-out',
    {
      pattern: /.*-square/, // colored squares
    },
    {
      pattern: /.*-chip/, // colored chips
    }
   ],
  theme: {
    extend: {
      colors: {
        primary: '#5865F2',           // Blurple, Discord's brand color
        secondary: colors.sky[400],
        accent: colors.purple[500],
        neutral: colors.slate[300],
        dark: colors.slate[800],
        discord: '#5865F2',           // Same color as primary
        'discord-darker': '#4F5AD9',  // Like blurple but darker
        'toast-default': colors.blue[800],
        'toast-error': colors.red[800],
      }
    }
  }
}