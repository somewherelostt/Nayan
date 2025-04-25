/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["../templates/**/*.html", "../static/js/**/*.js"],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: "#1d4ed8",
          light: "#3b82f6",
          dark: "#1e40af",
        },
        secondary: {
          DEFAULT: "#6d28d9",
          dark: "#5b21b6",
        },
        success: "#10b981",
        warning: "#f59e0b",
        danger: "#ef4444",
        dark: "#1f2937",
        light: "#f9fafb",
        "text-color": "#374151",
        "text-light": "#9ca3af",
        "border-color": "#e5e7eb",
        "bg-color": "#f3f4f6",
        "card-bg": "#ffffff",
      },
      animation: {
        scan: "scan 2s linear infinite",
        pulse: "pulse 2s infinite",
        fadeIn: "fadeIn 0.5s ease-out",
        slideIn: "slideIn 0.3s ease-out",
      },
      keyframes: {
        scan: {
          "0%": { top: "0" },
          "100%": { top: "100%" },
        },
        pulse: {
          "0%": { transform: "scale(1)" },
          "50%": { transform: "scale(1.2)" },
          "100%": { transform: "scale(1)" },
        },
        fadeIn: {
          from: { opacity: "0" },
          to: { opacity: "1" },
        },
        slideIn: {
          from: { transform: "translateY(20px)", opacity: "0" },
          to: { transform: "translateY(0)", opacity: "1" },
        },
      },
    },
  },
  plugins: [],
};
